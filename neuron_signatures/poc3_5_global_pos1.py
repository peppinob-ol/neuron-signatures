"""
POC 3.5: Global cross-layer steering at pos=-1

Key insight: Constraining to pos=-1 makes global search tractable.
- One backward pass captures gradients for ALL layers at the final token
- Rank 26*9216 = 239,616 candidates by delta*grad
- Patch in greedy batches to handle non-additivity

Design choices:
1. Freeze parameter grads (only need activation grads)
2. Capture all layers' hook_post activations with retain_grad()
3. Selection modes: per_layer quota, global top-K, or greedy batches
4. Strong controls: random, permute, negdelta, wrong-source

Run example:
    python -m neuron_signatures.poc3_5_global_pos1 \
      --prompt "The capital of the state containing Dallas is" \
      --source_prompt "The capital of the state containing San Francisco is" \
      --target_token " Sacramento" --alt_token " Austin" \
      --layer_window 0:25 --alpha 1.0 \
      --select_mode greedy --k_steps "1,2,5,10,20,50,100,200" \
      --controls "random,permute,negdelta" \
      --out_dir runs/poc3_5_global_test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from neuron_signatures.neuron_influence import (
    DTYPE_MAP,
    load_model,
    normalize_pos,
    resolve_single_token_id,
)
from neuron_signatures.token_sanitize import safe_print, sanitize_token


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class NeuronCandidate:
    """A single (layer, neuron) candidate for steering."""
    layer: int
    neuron_idx: int
    delta: float
    grad: float
    pred: float  # delta * grad

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "neuron_idx": self.neuron_idx,
            "delta": self.delta,
            "grad": self.grad,
            "pred": self.pred,
        }


@dataclass
class SteeringPlan:
    """Plan for multi-layer intervention at pos=-1."""
    pos: int
    candidates: List[NeuronCandidate] = field(default_factory=list)
    total_pred: float = 0.0

    def add(self, c: NeuronCandidate):
        self.candidates.append(c)
        self.total_pred += c.pred

    def by_layer(self) -> Dict[int, List[NeuronCandidate]]:
        """Group candidates by layer for hook application."""
        result: Dict[int, List[NeuronCandidate]] = {}
        for c in self.candidates:
            if c.layer not in result:
                result[c.layer] = []
            result[c.layer].append(c)
        return result


@dataclass
class LogitMetrics:
    """Comprehensive logit-based metrics."""
    logit_target: float
    logit_alt: float
    logit_diff: float  # target - alt
    rank_target: int
    rank_alt: int
    logit_top1: float
    top1_token: str
    top1_id: int
    prob_target: float
    prob_alt: float
    topk: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------------

def compute_logit_metrics(
    tokenizer,
    logits_1d: torch.Tensor,
    target_id: int,
    alt_id: int,
    topk_n: int = 10,
) -> LogitMetrics:
    """Compute comprehensive metrics from logits."""
    logits = logits_1d.float().cpu()

    lt = float(logits[target_id].item())
    la = float(logits[alt_id].item())

    rank_target = int((logits > lt).sum().item())
    rank_alt = int((logits > la).sum().item())

    top1_idx = int(logits.argmax().item())
    top1_logit = float(logits[top1_idx].item())
    top1_token = sanitize_token(tokenizer.decode([top1_idx]))

    probs = torch.softmax(logits, dim=-1)
    prob_target = float(probs[target_id].item())
    prob_alt = float(probs[alt_id].item())

    top_vals, top_idxs = torch.topk(logits, topk_n)
    topk = []
    for v, i in zip(top_vals.tolist(), top_idxs.tolist()):
        topk.append({
            "token": sanitize_token(tokenizer.decode([i])),
            "logit": float(v),
            "prob": float(probs[i].item()),
            "id": int(i),
        })

    return LogitMetrics(
        logit_target=lt,
        logit_alt=la,
        logit_diff=lt - la,
        rank_target=rank_target,
        rank_alt=rank_alt,
        logit_top1=top1_logit,
        top1_token=top1_token,
        top1_id=top1_idx,
        prob_target=prob_target,
        prob_alt=prob_alt,
        topk=topk,
    )


def metrics_to_dict(m: LogitMetrics) -> Dict[str, Any]:
    return {
        "logit_target": m.logit_target,
        "logit_alt": m.logit_alt,
        "logit_diff": m.logit_diff,
        "rank_target": m.rank_target,
        "rank_alt": m.rank_alt,
        "logit_top1": m.logit_top1,
        "top1_token": m.top1_token,
        "top1_id": m.top1_id,
        "prob_target": m.prob_target,
        "prob_alt": m.prob_alt,
        "topk": m.topk,
    }


# -----------------------------------------------------------------------------
# Activation capture with gradients (all layers, pos=-1)
# -----------------------------------------------------------------------------

def capture_acts_and_grads_dest(
    model,
    tokens: torch.Tensor,
    layers: List[int],
    pos: int,
    target_id: int,
    alt_id: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], float]:
    """
    Forward dest prompt with gradients, capture activations at all layers.

    Returns:
        acts: {layer: [d_mlp] tensor} - activations at pos
        grads: {layer: [d_mlp] tensor} - gradients at pos
        baseline_diff: logit_target - logit_alt
    """
    captured: Dict[int, torch.Tensor] = {}

    def make_capture(layer: int):
        def hook_fn(act: torch.Tensor, hook):
            act.retain_grad()
            captured[layer] = act
            return act
        return hook_fn

    hook_names = [(f"blocks.{L}.mlp.hook_post", make_capture(L)) for L in layers]

    model.reset_hooks(including_permanent=True)

    with torch.enable_grad():
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=hook_names,
            return_type="logits",
        )
        lt = logits[0, -1, target_id].float()
        la = logits[0, -1, alt_id].float()
        diff = lt - la
        diff.backward()

    model.reset_hooks(including_permanent=True)

    # Extract activations and grads at the target position
    acts = {}
    grads = {}
    for L in layers:
        if L not in captured:
            raise RuntimeError(f"Layer {L} activation not captured")
        act_full = captured[L]
        grad_full = act_full.grad
        if grad_full is None:
            raise RuntimeError(f"Layer {L} gradient is None")
        acts[L] = act_full[0, pos].detach().float().cpu()
        grads[L] = grad_full[0, pos].detach().float().cpu()

    baseline_diff = float(diff.detach().item())
    return acts, grads, baseline_diff


def capture_acts_src(
    model,
    tokens: torch.Tensor,
    layers: List[int],
    pos: int,
) -> Dict[int, torch.Tensor]:
    """
    Forward source prompt (no grad), capture activations at all layers.

    Returns:
        acts: {layer: [d_mlp] tensor} - activations at pos
    """
    captured: Dict[int, torch.Tensor] = {}

    def make_capture(layer: int):
        def hook_fn(act: torch.Tensor, hook):
            captured[layer] = act.detach()
            return act
        return hook_fn

    hook_names = [(f"blocks.{L}.mlp.hook_post", make_capture(L)) for L in layers]

    model.reset_hooks(including_permanent=True)

    with torch.inference_mode():
        _ = model.run_with_hooks(
            tokens,
            fwd_hooks=hook_names,
            return_type="logits",
        )

    model.reset_hooks(including_permanent=True)

    acts = {}
    for L in layers:
        if L not in captured:
            raise RuntimeError(f"Layer {L} activation not captured (src)")
        acts[L] = captured[L][0, pos].float().cpu()

    return acts


# -----------------------------------------------------------------------------
# Candidate ranking
# -----------------------------------------------------------------------------

def rank_candidates_pos1(
    acts_dest: Dict[int, torch.Tensor],
    acts_src: Dict[int, torch.Tensor],
    grads: Dict[int, torch.Tensor],
    layers: List[int],
    positive_only: bool = True,
) -> List[NeuronCandidate]:
    """
    Rank all (layer, neuron) candidates by predicted effect = delta * grad.

    Args:
        positive_only: If True, only include candidates with pred > 0
    """
    candidates = []

    for L in layers:
        delta = acts_src[L] - acts_dest[L]  # [d_mlp]
        grad = grads[L]  # [d_mlp]
        pred = delta * grad  # [d_mlp]

        d_mlp = delta.shape[0]
        for i in range(d_mlp):
            p = pred[i].item()
            if positive_only and p <= 0:
                continue
            candidates.append(NeuronCandidate(
                layer=L,
                neuron_idx=i,
                delta=delta[i].item(),
                grad=grad[i].item(),
                pred=p,
            ))

    # Sort by pred descending
    candidates.sort(key=lambda c: c.pred, reverse=True)
    return candidates


def select_per_layer(
    candidates: List[NeuronCandidate],
    k_per_layer: int,
) -> List[NeuronCandidate]:
    """Select top-k candidates per layer."""
    by_layer: Dict[int, List[NeuronCandidate]] = {}
    for c in candidates:
        if c.layer not in by_layer:
            by_layer[c.layer] = []
        by_layer[c.layer].append(c)

    selected = []
    for L in sorted(by_layer.keys()):
        layer_cands = by_layer[L]
        # Already sorted globally, but re-sort per layer to be safe
        layer_cands.sort(key=lambda c: c.pred, reverse=True)
        selected.extend(layer_cands[:k_per_layer])

    # Re-sort selected by pred
    selected.sort(key=lambda c: c.pred, reverse=True)
    return selected


def select_global_topk(
    candidates: List[NeuronCandidate],
    k_global: int,
) -> List[NeuronCandidate]:
    """Select top-k candidates globally."""
    return candidates[:k_global]


# -----------------------------------------------------------------------------
# Intervention hooks (multi-layer, pos=-1)
# -----------------------------------------------------------------------------

def make_multi_layer_hook(
    plan_by_layer: Dict[int, List[NeuronCandidate]],
    pos: int,
    alpha: float,
):
    """
    Create a hook function for a specific layer.

    Returns a factory that takes a layer and returns a hook for that layer.
    """
    def make_hook_for_layer(layer: int):
        if layer not in plan_by_layer:
            return None

        layer_candidates = plan_by_layer[layer]
        idxs = torch.tensor([c.neuron_idx for c in layer_candidates], dtype=torch.long)
        deltas = torch.tensor([c.delta for c in layer_candidates], dtype=torch.float32)

        def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
            # Patch in-place at selected indices
            d = (alpha * deltas).to(device=act.device, dtype=act.dtype)
            act[:, pos, idxs.to(act.device)] = act[:, pos, idxs.to(act.device)] + d
            return act

        return hook_fn

    return make_hook_for_layer


def run_with_plan(
    model,
    tokens: torch.Tensor,
    plan: SteeringPlan,
    alpha: float,
) -> torch.Tensor:
    """Run forward with multi-layer patching, return next-token logits."""
    plan_by_layer = plan.by_layer()
    hook_factory = make_multi_layer_hook(plan_by_layer, plan.pos, alpha)

    fwd_hooks = []
    for layer in plan_by_layer.keys():
        hook_fn = hook_factory(layer)
        if hook_fn is not None:
            fwd_hooks.append((f"blocks.{layer}.mlp.hook_post", hook_fn))

    model.reset_hooks(including_permanent=True)

    with torch.inference_mode():
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=fwd_hooks,
            return_type="logits",
        )

    model.reset_hooks(including_permanent=True)
    return logits[0, -1]


# -----------------------------------------------------------------------------
# Control interventions
# -----------------------------------------------------------------------------

def make_random_plan(
    n_neurons: int,
    layers: List[int],
    d_mlp: int,
    pos: int,
    delta_magnitude: float,
) -> SteeringPlan:
    """Random neurons with random deltas of given magnitude."""
    plan = SteeringPlan(pos=pos)
    for _ in range(n_neurons):
        L = random.choice(layers)
        i = random.randint(0, d_mlp - 1)
        # Random sign
        d = delta_magnitude * (1.0 if random.random() > 0.5 else -1.0)
        plan.add(NeuronCandidate(layer=L, neuron_idx=i, delta=d, grad=0.0, pred=0.0))
    return plan


def make_permuted_plan(
    original_plan: SteeringPlan,
) -> SteeringPlan:
    """Permute deltas across neurons within each layer."""
    plan_by_layer = original_plan.by_layer()
    new_plan = SteeringPlan(pos=original_plan.pos)

    for L, cands in plan_by_layer.items():
        # Extract deltas and shuffle
        deltas = [c.delta for c in cands]
        random.shuffle(deltas)
        # Create new candidates with shuffled deltas
        for c, new_delta in zip(cands, deltas):
            new_plan.add(NeuronCandidate(
                layer=c.layer,
                neuron_idx=c.neuron_idx,
                delta=new_delta,
                grad=c.grad,
                pred=0.0,  # Not meaningful after permutation
            ))

    return new_plan


def make_negdelta_plan(
    original_plan: SteeringPlan,
) -> SteeringPlan:
    """Negate all deltas (should hurt objective if method is valid)."""
    new_plan = SteeringPlan(pos=original_plan.pos)
    for c in original_plan.candidates:
        new_plan.add(NeuronCandidate(
            layer=c.layer,
            neuron_idx=c.neuron_idx,
            delta=-c.delta,
            grad=c.grad,
            pred=-c.pred,
        ))
    return new_plan


# -----------------------------------------------------------------------------
# Greedy batched selection
# -----------------------------------------------------------------------------

def run_greedy_batches(
    model,
    tokens: torch.Tensor,
    candidates: List[NeuronCandidate],
    pos: int,
    alpha: float,
    k_steps: List[int],
    target_id: int,
    alt_id: int,
    baseline: LogitMetrics,
    topk_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Run greedy batched intervention: try K=1,2,5,10,... neurons.

    Returns list of results per K step.
    """
    results = []

    for k in k_steps:
        if k > len(candidates):
            k = len(candidates)
        if k == 0:
            continue

        # Build plan with top-k candidates
        plan = SteeringPlan(pos=pos)
        for c in candidates[:k]:
            plan.add(c)

        # Run intervention
        steered_logits = run_with_plan(model, tokens, plan, alpha)
        steered = compute_logit_metrics(model.tokenizer, steered_logits, target_id, alt_id, topk_n)

        d_logit_diff = steered.logit_diff - baseline.logit_diff
        d_target_logit = steered.logit_target - baseline.logit_target
        d_alt_logit = steered.logit_alt - baseline.logit_alt

        # Count layers involved
        layers_involved = list(set(c.layer for c in candidates[:k]))

        results.append({
            "k": k,
            "n_layers": len(layers_involved),
            "layers": sorted(layers_involved),
            "pred_sum": plan.total_pred * alpha,
            "d_logit_diff": d_logit_diff,
            "d_target_logit": d_target_logit,
            "d_alt_logit": d_alt_logit,
            "pred_vs_actual": (plan.total_pred * alpha) / d_logit_diff if abs(d_logit_diff) > 1e-6 else float('nan'),
            "steered": metrics_to_dict(steered),
            "success": steered.rank_target <= steered.rank_alt,  # Target outranks alt
        })

    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="POC 3.5: Global cross-layer steering at pos=-1")

    # Model
    ap.add_argument("--model_name", default="google/gemma-2-2b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--no_prepend_bos", action="store_true")

    # Prompts
    ap.add_argument("--prompt", required=True, help="Destination prompt")
    ap.add_argument("--source_prompt", required=True, help="Source prompt (donor activations)")
    ap.add_argument("--wrong_source_prompt", default=None, help="Wrong-source control prompt")

    # Tokens
    ap.add_argument("--target_token", required=True, help="Token to promote (e.g., ' Sacramento')")
    ap.add_argument("--alt_token", required=True, help="Token to compare against (e.g., ' Austin')")

    # Position (fixed to -1 for this POC)
    ap.add_argument("--pos_mode", default="last", choices=["last"])
    ap.add_argument("--pos", type=int, default=-1)

    # Layer window
    ap.add_argument("--layer_window", default="0:25", help="Layer range, e.g., '0:25' or '18:22'")

    # Intervention
    ap.add_argument("--alpha", type=float, default=1.0)

    # Selection
    ap.add_argument("--select_mode", default="greedy", choices=["per_layer", "global", "greedy"])
    ap.add_argument("--k_per_layer", type=int, default=10, help="For per_layer mode")
    ap.add_argument("--k_global", type=int, default=200, help="For global mode")
    ap.add_argument("--k_steps", default="1,2,5,10,20,50,100,200", help="For greedy mode, comma-separated K values")
    ap.add_argument("--positive_only", action="store_true", default=True,
                    help="Only select neurons with positive pred (default: True)")
    ap.add_argument("--include_negative", action="store_true",
                    help="Include neurons with negative pred in ranking")

    # Controls
    ap.add_argument("--controls", default="random,permute,negdelta",
                    help="Comma-separated controls: random, permute, negdelta, wrongsrc")
    ap.add_argument("--control_trials", type=int, default=5, help="Number of trials for stochastic controls")

    # Output
    ap.add_argument("--topk_tokens", type=int, default=10)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Parse layer window
    layer_parts = args.layer_window.split(":")
    layer_start = int(layer_parts[0])
    layer_end = int(layer_parts[1]) if len(layer_parts) > 1 else layer_start
    layers = list(range(layer_start, layer_end + 1))

    # Parse k_steps
    k_steps = [int(x.strip()) for x in args.k_steps.split(",")]

    # Parse controls
    controls = [c.strip() for c in args.controls.split(",") if c.strip()]

    dtype = DTYPE_MAP[args.dtype]
    prepend_bos = not args.no_prepend_bos
    positive_only = args.positive_only and not args.include_negative

    safe_print("=" * 70)
    safe_print("POC 3.5: Global Cross-Layer Steering at pos=-1")
    safe_print("=" * 70)
    safe_print(f"Model: {args.model_name} | dtype: {args.dtype} | seed: {args.seed}")
    safe_print(f"Layers: {layer_start}-{layer_end} ({len(layers)} layers)")
    safe_print(f"Dest: {args.prompt}")
    safe_print(f"Src:  {args.source_prompt}")
    safe_print(f"Target: {args.target_token!r} | Alt: {args.alt_token!r}")
    safe_print(f"Selection: {args.select_mode} | alpha: {args.alpha}")
    safe_print(f"Controls: {controls}")
    safe_print("")

    # Load model
    safe_print("Loading model...")
    model = load_model(args.model_name, args.device, dtype)
    d_mlp = model.cfg.d_mlp

    # IMPORTANT: Freeze parameters to avoid allocating .grad buffers for weights
    safe_print("Freezing model parameters...")
    for p in model.parameters():
        p.requires_grad_(False)

    target_id = resolve_single_token_id(model, args.target_token)
    alt_id = resolve_single_token_id(model, args.alt_token)

    safe_print(f"Target: {args.target_token!r} -> id={target_id}")
    safe_print(f"Alt: {args.alt_token!r} -> id={alt_id}")
    safe_print("")

    # Tokenize
    dest_tokens = model.to_tokens(args.prompt, prepend_bos=prepend_bos)
    src_tokens = model.to_tokens(args.source_prompt, prepend_bos=prepend_bos)

    dest_seq_len = dest_tokens.shape[1]
    src_seq_len = src_tokens.shape[1]
    pos_dest = normalize_pos(args.pos, dest_seq_len)
    pos_src = normalize_pos(args.pos, src_seq_len)

    safe_print(f"Dest seq_len: {dest_seq_len}, pos: {pos_dest}")
    safe_print(f"Src seq_len: {src_seq_len}, pos: {pos_src}")

    # Show token at pos
    dest_tok_at_pos = model.tokenizer.decode([dest_tokens[0, pos_dest].item()])
    src_tok_at_pos = model.tokenizer.decode([src_tokens[0, pos_src].item()])
    safe_print(f"Token at dest pos={pos_dest}: {sanitize_token(dest_tok_at_pos)!r}")
    safe_print(f"Token at src pos={pos_src}: {sanitize_token(src_tok_at_pos)!r}")
    safe_print("")

    # =========================================================================
    # Step 1: Capture activations and gradients
    # =========================================================================
    safe_print("Step 1: Capturing activations and gradients (dest with grad, src no grad)...")

    acts_dest, grads, baseline_diff = capture_acts_and_grads_dest(
        model, dest_tokens, layers, pos_dest, target_id, alt_id
    )
    safe_print(f"  Baseline logit_diff (target-alt): {baseline_diff:+.4f}")

    acts_src = capture_acts_src(model, src_tokens, layers, pos_src)
    safe_print(f"  Captured activations for {len(layers)} layers")

    # =========================================================================
    # Step 2: Rank candidates
    # =========================================================================
    safe_print("\nStep 2: Ranking candidates by delta*grad...")

    candidates = rank_candidates_pos1(acts_dest, acts_src, grads, layers, positive_only=positive_only)
    safe_print(f"  Total candidates (pred>0): {len(candidates)}")

    if len(candidates) == 0:
        safe_print("ERROR: No candidates with positive pred. Try --include_negative")
        return

    # Show top candidates
    safe_print("\n  Top 20 candidates:")
    for i, c in enumerate(candidates[:20]):
        safe_print(f"    [{i+1:2d}] L{c.layer:2d}.{c.neuron_idx:5d}: pred={c.pred:+.4f} (d={c.delta:+.3f}, g={c.grad:+.4f})")

    # Layer distribution
    layer_counts = {}
    layer_pred_mass = {}
    for c in candidates:
        layer_counts[c.layer] = layer_counts.get(c.layer, 0) + 1
        layer_pred_mass[c.layer] = layer_pred_mass.get(c.layer, 0.0) + c.pred

    safe_print("\n  Per-layer candidate distribution (top 10 by pred mass):")
    sorted_layers = sorted(layer_pred_mass.keys(), key=lambda L: layer_pred_mass[L], reverse=True)
    for L in sorted_layers[:10]:
        safe_print(f"    L{L:2d}: {layer_counts[L]:5d} candidates, pred_mass={layer_pred_mass[L]:+.4f}")

    # =========================================================================
    # Step 3: Baseline metrics
    # =========================================================================
    safe_print("\nStep 3: Computing baseline metrics...")

    with torch.inference_mode():
        base_logits = model(dest_tokens, return_type="logits")[0, -1]
    baseline = compute_logit_metrics(model.tokenizer, base_logits, target_id, alt_id, args.topk_tokens)

    safe_print(f"  Baseline:")
    safe_print(f"    logit_target={baseline.logit_target:+.4f}, logit_alt={baseline.logit_alt:+.4f}")
    safe_print(f"    logit_diff={baseline.logit_diff:+.4f}")
    safe_print(f"    rank_target={baseline.rank_target}, rank_alt={baseline.rank_alt}")
    safe_print(f"    top1: {baseline.top1_token!r} (logit={baseline.logit_top1:.4f})")

    # =========================================================================
    # Step 4: Selection and intervention
    # =========================================================================
    safe_print(f"\nStep 4: Running {args.select_mode} selection + intervention...")

    main_results = []

    if args.select_mode == "per_layer":
        selected = select_per_layer(candidates, args.k_per_layer)
        safe_print(f"  Selected {len(selected)} neurons ({args.k_per_layer} per layer)")

        plan = SteeringPlan(pos=pos_dest)
        for c in selected:
            plan.add(c)

        steered_logits = run_with_plan(model, dest_tokens, plan, args.alpha)
        steered = compute_logit_metrics(model.tokenizer, steered_logits, target_id, alt_id, args.topk_tokens)

        d_logit_diff = steered.logit_diff - baseline.logit_diff
        main_results.append({
            "mode": "per_layer",
            "k": len(selected),
            "k_per_layer": args.k_per_layer,
            "pred_sum": plan.total_pred * args.alpha,
            "d_logit_diff": d_logit_diff,
            "pred_vs_actual": (plan.total_pred * args.alpha) / d_logit_diff if abs(d_logit_diff) > 1e-6 else float('nan'),
            "steered": metrics_to_dict(steered),
        })

    elif args.select_mode == "global":
        selected = select_global_topk(candidates, args.k_global)
        safe_print(f"  Selected {len(selected)} neurons (global top-{args.k_global})")

        plan = SteeringPlan(pos=pos_dest)
        for c in selected:
            plan.add(c)

        steered_logits = run_with_plan(model, dest_tokens, plan, args.alpha)
        steered = compute_logit_metrics(model.tokenizer, steered_logits, target_id, alt_id, args.topk_tokens)

        d_logit_diff = steered.logit_diff - baseline.logit_diff
        main_results.append({
            "mode": "global",
            "k": len(selected),
            "pred_sum": plan.total_pred * args.alpha,
            "d_logit_diff": d_logit_diff,
            "pred_vs_actual": (plan.total_pred * args.alpha) / d_logit_diff if abs(d_logit_diff) > 1e-6 else float('nan'),
            "steered": metrics_to_dict(steered),
        })

    elif args.select_mode == "greedy":
        safe_print(f"  Running greedy batches: K = {k_steps}")
        main_results = run_greedy_batches(
            model, dest_tokens, candidates, pos_dest, args.alpha,
            k_steps, target_id, alt_id, baseline, args.topk_tokens
        )

        safe_print("\n  Greedy results:")
        for r in main_results:
            status = "SUCCESS" if r["success"] else ""
            safe_print(f"    K={r['k']:4d}: d_diff={r['d_logit_diff']:+.4f}, pred={r['pred_sum']:+.4f}, "
                      f"p/a={r['pred_vs_actual']:.2f}, layers={r['n_layers']} {status}")

    # Determine "best" result for controls
    if args.select_mode == "greedy":
        # Use the K that achieves success, or largest K
        best_idx = len(main_results) - 1
        for i, r in enumerate(main_results):
            if r["success"]:
                best_idx = i
                break
        best_k = main_results[best_idx]["k"]
    else:
        best_k = len(selected) if args.select_mode != "greedy" else k_steps[-1]

    # Build the "best" plan for controls
    best_plan = SteeringPlan(pos=pos_dest)
    for c in candidates[:best_k]:
        best_plan.add(c)

    # =========================================================================
    # Step 5: Controls
    # =========================================================================
    control_results = {}

    if "random" in controls:
        safe_print(f"\nControl: Random neurons (n={args.control_trials} trials)...")
        random_diffs = []
        avg_delta_mag = sum(abs(c.delta) for c in candidates[:best_k]) / best_k if best_k > 0 else 1.0

        for t in range(args.control_trials):
            rand_plan = make_random_plan(best_k, layers, d_mlp, pos_dest, avg_delta_mag)
            rand_logits = run_with_plan(model, dest_tokens, rand_plan, args.alpha)
            rand_m = compute_logit_metrics(model.tokenizer, rand_logits, target_id, alt_id, args.topk_tokens)
            random_diffs.append(rand_m.logit_diff - baseline.logit_diff)

        avg_rand = sum(random_diffs) / len(random_diffs)
        std_rand = (sum((x - avg_rand)**2 for x in random_diffs) / len(random_diffs)) ** 0.5
        main_d_diff = main_results[-1]["d_logit_diff"] if main_results else 0.0
        control_results["random"] = {
            "d_logit_diff_mean": avg_rand,
            "d_logit_diff_std": std_rand,
            "samples": random_diffs,
            "specificity": abs(main_d_diff) / max(abs(avg_rand), 1e-6),
        }
        safe_print(f"  Random: d_diff_mean={avg_rand:+.4f} +/- {std_rand:.4f}, specificity={control_results['random']['specificity']:.1f}x")

    if "permute" in controls:
        safe_print(f"\nControl: Permuted deltas (n={args.control_trials} trials)...")
        perm_diffs = []

        for t in range(args.control_trials):
            perm_plan = make_permuted_plan(best_plan)
            perm_logits = run_with_plan(model, dest_tokens, perm_plan, args.alpha)
            perm_m = compute_logit_metrics(model.tokenizer, perm_logits, target_id, alt_id, args.topk_tokens)
            perm_diffs.append(perm_m.logit_diff - baseline.logit_diff)

        avg_perm = sum(perm_diffs) / len(perm_diffs)
        std_perm = (sum((x - avg_perm)**2 for x in perm_diffs) / len(perm_diffs)) ** 0.5
        main_d_diff = main_results[-1]["d_logit_diff"] if main_results else 0.0
        control_results["permute"] = {
            "d_logit_diff_mean": avg_perm,
            "d_logit_diff_std": std_perm,
            "samples": perm_diffs,
            "specificity": abs(main_d_diff) / max(abs(avg_perm), 1e-6),
        }
        safe_print(f"  Permute: d_diff_mean={avg_perm:+.4f} +/- {std_perm:.4f}, specificity={control_results['permute']['specificity']:.1f}x")

    if "negdelta" in controls:
        safe_print("\nControl: Negated deltas...")
        neg_plan = make_negdelta_plan(best_plan)
        neg_logits = run_with_plan(model, dest_tokens, neg_plan, args.alpha)
        neg_m = compute_logit_metrics(model.tokenizer, neg_logits, target_id, alt_id, args.topk_tokens)
        neg_d_diff = neg_m.logit_diff - baseline.logit_diff

        main_d_diff = main_results[-1]["d_logit_diff"] if main_results else 0.0
        control_results["negdelta"] = {
            "d_logit_diff": neg_d_diff,
            "direction_correct": (main_d_diff > 0 and neg_d_diff < 0) or (main_d_diff < 0 and neg_d_diff > 0),
            "steered": metrics_to_dict(neg_m),
        }
        safe_print(f"  Negdelta: d_diff={neg_d_diff:+.4f} (direction_correct={control_results['negdelta']['direction_correct']})")

    if "wrongsrc" in controls and args.wrong_source_prompt:
        safe_print("\nControl: Wrong source prompt...")
        wrong_tokens = model.to_tokens(args.wrong_source_prompt, prepend_bos=prepend_bos)
        pos_wrong = normalize_pos(args.pos, wrong_tokens.shape[1])
        acts_wrong = capture_acts_src(model, wrong_tokens, layers, pos_wrong)

        # Recompute deltas with wrong source
        wrong_candidates = rank_candidates_pos1(acts_dest, acts_wrong, grads, layers, positive_only=positive_only)
        wrong_plan = SteeringPlan(pos=pos_dest)
        for c in wrong_candidates[:best_k]:
            wrong_plan.add(c)

        wrong_logits = run_with_plan(model, dest_tokens, wrong_plan, args.alpha)
        wrong_m = compute_logit_metrics(model.tokenizer, wrong_logits, target_id, alt_id, args.topk_tokens)
        wrong_d_diff = wrong_m.logit_diff - baseline.logit_diff

        main_d_diff = main_results[-1]["d_logit_diff"] if main_results else 0.0
        control_results["wrongsrc"] = {
            "d_logit_diff": wrong_d_diff,
            "specificity": abs(main_d_diff) / max(abs(wrong_d_diff), 1e-6),
            "steered": metrics_to_dict(wrong_m),
        }
        safe_print(f"  WrongSrc: d_diff={wrong_d_diff:+.4f}, specificity={control_results['wrongsrc']['specificity']:.1f}x")

    # =========================================================================
    # Step 6: Summary
    # =========================================================================
    safe_print("\n" + "=" * 70)
    safe_print("SUMMARY")
    safe_print("=" * 70)

    safe_print(f"\nBaseline: logit_diff={baseline.logit_diff:+.4f}, rank_target={baseline.rank_target}, top1={baseline.top1_token!r}")

    if main_results:
        best_r = main_results[-1]
        safe_print(f"\nBest intervention (K={best_r.get('k', best_k)}):")
        safe_print(f"  d_logit_diff = {best_r['d_logit_diff']:+.4f}")
        safe_print(f"  pred/actual  = {best_r['pred_vs_actual']:.2f}")
        if "steered" in best_r:
            s = best_r["steered"]
            safe_print(f"  steered: logit_diff={s['logit_diff']:+.4f}, rank_target={s['rank_target']}, top1={s['top1_token']!r}")

        if best_r.get("success"):
            safe_print("  STATUS: SUCCESS (target outranks alt)")
        else:
            safe_print("  STATUS: Target does not outrank alt yet")

    safe_print("\nControls:")
    for ctrl_name, ctrl_data in control_results.items():
        if "specificity" in ctrl_data:
            safe_print(f"  {ctrl_name}: specificity = {ctrl_data['specificity']:.1f}x")
        elif "direction_correct" in ctrl_data:
            safe_print(f"  {ctrl_name}: direction_correct = {ctrl_data['direction_correct']}")

    # =========================================================================
    # Step 7: Save outputs
    # =========================================================================
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"poc3_5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    report = {
        "created_at_utc": _now_utc(),
        "model_name": args.model_name,
        "dtype": args.dtype,
        "seed": args.seed,
        "prompt": args.prompt,
        "source_prompt": args.source_prompt,
        "wrong_source_prompt": args.wrong_source_prompt,
        "target_token": args.target_token,
        "target_id": target_id,
        "alt_token": args.alt_token,
        "alt_id": alt_id,
        "pos": pos_dest,
        "layers": layers,
        "alpha": args.alpha,
        "select_mode": args.select_mode,
        "k_per_layer": args.k_per_layer,
        "k_global": args.k_global,
        "k_steps": k_steps,
        "total_candidates": len(candidates),
        "baseline": metrics_to_dict(baseline),
        "main_results": main_results,
        "control_results": control_results,
        "layer_distribution": {
            "counts": {str(k): v for k, v in layer_counts.items()},
            "pred_mass": {str(k): v for k, v in layer_pred_mass.items()},
        },
    }

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    safe_print(f"\nSaved: {report_path}")

    # Save top candidates CSV
    csv_path = out_dir / "candidates_top500.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "layer", "neuron_idx", "delta", "grad", "pred"])
        writer.writeheader()
        for i, c in enumerate(candidates[:500]):
            writer.writerow({
                "rank": i + 1,
                "layer": c.layer,
                "neuron_idx": c.neuron_idx,
                "delta": f"{c.delta:.6f}",
                "grad": f"{c.grad:.6f}",
                "pred": f"{c.pred:.6f}",
            })
    safe_print(f"Saved: {csv_path}")

    safe_print("\nDone.")


if __name__ == "__main__":
    main()

