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
4. Strong controls: random, permute, permute_cross, negdelta, wrongpos, wrongsrc
5. Dual metrics: track both target-alt AND target-objective_alt when objective differs

Run example:
    python -m neuron_signatures.poc3_5_global_pos1 \\
      --prompt "The capital of the state containing Dallas is" \\
      --source_prompt "The capital of the state containing San Francisco is" \\
      --target_token " Sacramento" --alt_token " Austin" \\
      --layer_window 0:25 --alpha 1.0 \\
      --select_mode greedy --k_steps "1,2,5,10,20,50,100,200" \\
      --controls "random,permute,negdelta,wrongpos" \\
      --out_dir runs/poc3_5_global_test
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

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

    def with_new_pos(self, new_pos: int) -> "SteeringPlan":
        """Create a copy with a different position (for wrongpos control)."""
        new_plan = SteeringPlan(pos=new_pos)
        for c in self.candidates:
            new_plan.add(NeuronCandidate(
                layer=c.layer,
                neuron_idx=c.neuron_idx,
                delta=c.delta,
                grad=c.grad,
                pred=c.pred,
            ))
        return new_plan


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


@dataclass
class DualMetrics:
    """Dual metrics: target-alt AND target-objective_alt."""
    # Primary: target vs alt (for final evaluation)
    target_alt: LogitMetrics
    # Objective: what we optimized for (may differ in target_top1 mode)
    target_obj: LogitMetrics
    # Are they the same?
    same_objective: bool


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


def compute_dual_metrics(
    tokenizer,
    logits_1d: torch.Tensor,
    target_id: int,
    alt_id: int,
    objective_alt_id: int,
    topk_n: int = 10,
) -> DualMetrics:
    """Compute dual metrics: both target-alt and target-objective_alt."""
    target_alt = compute_logit_metrics(tokenizer, logits_1d, target_id, alt_id, topk_n)

    same_objective = (alt_id == objective_alt_id)
    if same_objective:
        target_obj = target_alt
    else:
        target_obj = compute_logit_metrics(tokenizer, logits_1d, target_id, objective_alt_id, topk_n)

    return DualMetrics(
        target_alt=target_alt,
        target_obj=target_obj,
        same_objective=same_objective,
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


def dual_metrics_to_dict(dm: DualMetrics) -> Dict[str, Any]:
    return {
        "target_alt": metrics_to_dict(dm.target_alt),
        "target_obj": metrics_to_dict(dm.target_obj),
        "same_objective": dm.same_objective,
    }


# -----------------------------------------------------------------------------
# Activation capture with gradients (all layers, pos=-1)
# -----------------------------------------------------------------------------

def get_baseline_top1_id(
    model,
    tokens: torch.Tensor,
) -> int:
    """Get the top-1 predicted token id for baseline (no intervention)."""
    with torch.inference_mode():
        logits = model(tokens, return_type="logits")
    top1_id = int(logits[0, -1].argmax().item())
    return top1_id


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
        acts: {layer: [d_mlp] tensor} - activations at pos (CPU, float32)
        grads: {layer: [d_mlp] tensor} - gradients at pos (CPU, float32)
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

    # Extract activations and grads at the target position, move to CPU
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

    # Memory cleanup: release captured tensors
    del captured
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

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
        acts: {layer: [d_mlp] tensor} - activations at pos (CPU, float32)
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

    # Memory cleanup
    del captured
    gc.collect()

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
# Intervention hooks (multi-layer, pos=-1) - with device caching
# -----------------------------------------------------------------------------

def make_multi_layer_hook(
    plan_by_layer: Dict[int, List[NeuronCandidate]],
    pos: int,
    alpha: float,
):
    """
    Create a hook function factory for multi-layer intervention.

    Returns a factory that takes a layer and returns a hook for that layer.
    Uses lazy caching to avoid recreating tensors on every forward pass.
    """
    def make_hook_for_layer(layer: int):
        if layer not in plan_by_layer:
            return None

        layer_candidates = plan_by_layer[layer]
        idxs = torch.tensor([c.neuron_idx for c in layer_candidates], dtype=torch.long)
        deltas = torch.tensor([c.delta for c in layer_candidates], dtype=torch.float32)

        # Cached device tensors (lazy init)
        cache: Dict[str, Optional[torch.Tensor]] = {"idxs_dev": None, "deltas_dev": None}

        def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
            # Lazy caching: move to device only once (with explicit None guards)
            if (cache["idxs_dev"] is None or
                cache["deltas_dev"] is None or
                cache["idxs_dev"].device != act.device or
                cache["deltas_dev"].dtype != act.dtype):
                cache["idxs_dev"] = idxs.to(act.device)
                cache["deltas_dev"] = (alpha * deltas).to(device=act.device, dtype=act.dtype)

            # In-place addition at selected indices (torch-friendly)
            act[:, pos, cache["idxs_dev"]] += cache["deltas_dev"]
            return act

        return hook_fn

    return make_hook_for_layer


def run_with_plan(
    model,
    tokens: torch.Tensor,
    plan: SteeringPlan,
    alpha: float,
    plan_by_layer: Optional[Dict[int, List[NeuronCandidate]]] = None,
) -> torch.Tensor:
    """Run forward with multi-layer patching, return next-token logits.
    
    Args:
        plan_by_layer: Pre-computed by_layer dict for performance (optional).
    """
    if plan_by_layer is None:
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
# Control interventions (improved)
# -----------------------------------------------------------------------------

def make_random_plan_like(
    original_plan: SteeringPlan,
    layers: List[int],
    d_mlp: int,
) -> SteeringPlan:
    """
    Random neurons with SAME deltas as original (preserves delta budget).

    This is "random mapping, same delta budget" - a stronger control than
    random magnitude.
    """
    plan = SteeringPlan(pos=original_plan.pos)
    deltas = [c.delta for c in original_plan.candidates]
    random.shuffle(deltas)

    for d in deltas:
        L = random.choice(layers)
        i = random.randint(0, d_mlp - 1)
        plan.add(NeuronCandidate(layer=L, neuron_idx=i, delta=d, grad=0.0, pred=0.0))

    return plan


def make_permuted_plan_within_layer(
    original_plan: SteeringPlan,
) -> SteeringPlan:
    """Permute deltas across neurons WITHIN each layer (preserves layer structure)."""
    plan_by_layer = original_plan.by_layer()
    new_plan = SteeringPlan(pos=original_plan.pos)

    for L, cands in plan_by_layer.items():
        # Extract deltas and shuffle within layer
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


def make_permuted_plan_cross_layer(
    original_plan: SteeringPlan,
) -> SteeringPlan:
    """Permute deltas ACROSS layers (strongest permutation control)."""
    new_plan = SteeringPlan(pos=original_plan.pos)

    # Extract all deltas and shuffle globally
    all_deltas = [c.delta for c in original_plan.candidates]
    random.shuffle(all_deltas)

    # Assign shuffled deltas to original (layer, neuron) positions
    for c, new_delta in zip(original_plan.candidates, all_deltas):
        new_plan.add(NeuronCandidate(
            layer=c.layer,
            neuron_idx=c.neuron_idx,
            delta=new_delta,
            grad=c.grad,
            pred=0.0,
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
# Greedy batched selection (with dual metrics)
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
    objective_alt_id: int,
    baseline_dual: DualMetrics,
    topk_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Run greedy batched intervention: try K=1,2,5,10,... neurons.

    Returns list of results per K step with dual metrics.
    """
    results = []
    eps = 1e-6

    for k in k_steps:
        if k > len(candidates):
            k = len(candidates)
        if k == 0:
            continue

        # Build plan with top-k candidates
        plan = SteeringPlan(pos=pos)
        for c in candidates[:k]:
            plan.add(c)

        # Pre-compute plan_by_layer for performance
        plan_by_layer = plan.by_layer()

        # Run intervention
        steered_logits = run_with_plan(model, tokens, plan, alpha, plan_by_layer)
        steered_dual = compute_dual_metrics(model.tokenizer, steered_logits, target_id, alt_id, objective_alt_id, topk_n)

        # Compute deltas for BOTH metrics
        d_logit_diff_alt = steered_dual.target_alt.logit_diff - baseline_dual.target_alt.logit_diff
        d_target_logit = steered_dual.target_alt.logit_target - baseline_dual.target_alt.logit_target
        d_alt_logit = steered_dual.target_alt.logit_alt - baseline_dual.target_alt.logit_alt

        d_logit_diff_obj = steered_dual.target_obj.logit_diff - baseline_dual.target_obj.logit_diff

        pred_sum = plan.total_pred * alpha

        # Robust pred/actual metrics (on objective)
        pred_vs_actual = pred_sum / d_logit_diff_obj if abs(d_logit_diff_obj) > eps else float('nan')
        pred_minus_actual = pred_sum - d_logit_diff_obj

        # Count layers involved
        layers_involved = list(set(c.layer for c in candidates[:k]))

        results.append({
            "k": k,
            "n_layers": len(layers_involved),
            "layers": sorted(layers_involved),
            "pred_sum": pred_sum,
            # Primary (target vs alt)
            "d_logit_diff": d_logit_diff_alt,
            "d_target_logit": d_target_logit,
            "d_alt_logit": d_alt_logit,
            # Objective (what we optimized)
            "d_logit_diff_obj": d_logit_diff_obj,
            "pred_vs_actual": pred_vs_actual,
            "pred_minus_actual": pred_minus_actual,
            # Full steered metrics
            "steered": dual_metrics_to_dict(steered_dual),
            # Success: target outranks alt in primary metric
            "success": steered_dual.target_alt.rank_target <= steered_dual.target_alt.rank_alt,
        })

    return results


# -----------------------------------------------------------------------------
# Plotting (paper-grade figures)
# -----------------------------------------------------------------------------

def generate_plots(
    out_dir: Path,
    main_results: List[Dict[str, Any]],
    control_results: Dict[str, Any],
    baseline_logit_diff: float,
    main_effect: float,
    best_idx: int,
):
    """Generate standard paper-grade plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        safe_print("WARNING: matplotlib not available, skipping plot generation")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Plot 1: d_logit_diff vs K (with pred_sum overlay)
    # -------------------------------------------------------------------------
    if main_results and len(main_results) > 1:
        ks = [r["k"] for r in main_results]
        d_diffs = [r["d_logit_diff"] for r in main_results]
        pred_sums = [r["pred_sum"] for r in main_results]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = '#2E86AB'
        ax1.set_xlabel('K (number of neurons)', fontsize=12)
        ax1.set_ylabel('d_logit_diff (actual)', color=color1, fontsize=12)
        ax1.plot(ks, d_diffs, 'o-', color=color1, linewidth=2, markersize=6, label='Actual')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Zero-crossing threshold: delta needed to make logit_diff = 0
        zero_crossing = -baseline_logit_diff
        ax1.axhline(y=zero_crossing, color='red', linestyle=':', alpha=0.7, 
                   label=f'Zero-crossing ({zero_crossing:+.2f})')

        ax2 = ax1.twinx()
        color2 = '#A23B72'
        ax2.set_ylabel('pred_sum (predicted)', color=color2, fontsize=12)
        ax2.plot(ks, pred_sums, 's--', color=color2, linewidth=2, markersize=5, label='Predicted')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Mark best_idx point (the one used for controls comparison)
        best_k = main_results[best_idx]["k"]
        ax1.axvline(x=best_k, color='green', linestyle='-', alpha=0.3, linewidth=8, 
                   label=f'Best K={best_k}')

        fig.suptitle('Steering Effect vs Number of Neurons', fontsize=14)
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        fig.savefig(plots_dir / "greedy_curve.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        safe_print(f"  Saved: {plots_dir / 'greedy_curve.png'}")

    # -------------------------------------------------------------------------
    # Plot 2: Control comparison (boxplot/bar)
    # -------------------------------------------------------------------------
    control_names = []
    control_means = []
    control_stds = []
    # Use main_effect passed from caller (consistent with best_idx)

    for name, data in control_results.items():
        if "d_logit_diff_mean" in data:
            control_names.append(name)
            control_means.append(data["d_logit_diff_mean"])
            control_stds.append(data.get("d_logit_diff_std", 0.0))
        elif "d_logit_diff" in data and name != "negdelta":
            control_names.append(name)
            control_means.append(data["d_logit_diff"])
            control_stds.append(0.0)

    if control_names:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(len(control_names) + 1)
        all_names = ["MAIN"] + control_names
        all_means = [main_effect] + control_means
        all_stds = [0.0] + control_stds

        colors = ['#2E86AB'] + ['#A23B72'] * len(control_names)
        bars = ax.bar(x, all_means, yerr=all_stds, capsize=5, color=colors, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(all_names, rotation=45, ha='right')
        ax.set_ylabel('d_logit_diff', fontsize=12)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Main Effect vs Controls', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)

        # Add specificity annotations
        for i, name in enumerate(control_names):
            if "specificity" in control_results[name]:
                spec = control_results[name]["specificity"]
                ax.annotate(f'{spec:.1f}x', xy=(i+1, all_means[i+1]),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=9, color='#666')

        fig.tight_layout()
        fig.savefig(plots_dir / "controls_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        safe_print(f"  Saved: {plots_dir / 'controls_comparison.png'}")

    # -------------------------------------------------------------------------
    # Plot 3: Control samples distribution (if available)
    # -------------------------------------------------------------------------
    sample_data = {}
    for name, data in control_results.items():
        if "samples" in data:
            sample_data[name] = data["samples"]

    if sample_data and main_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        positions = []
        labels = []
        all_samples = []

        pos = 1
        for name, samples in sample_data.items():
            positions.append(pos)
            labels.append(name)
            all_samples.append(samples)
            pos += 1

        bp = ax.boxplot(all_samples, positions=positions, widths=0.6, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#A23B72')
            patch.set_alpha(0.6)

        # Add main effect line
        ax.axhline(y=main_effect, color='#2E86AB', linewidth=2, label=f'Main effect: {main_effect:.3f}')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('d_logit_diff', fontsize=12)
        ax.set_title('Control Distributions vs Main Effect', fontsize=14)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        fig.tight_layout()
        fig.savefig(plots_dir / "controls_distribution.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        safe_print(f"  Saved: {plots_dir / 'controls_distribution.png'}")


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

    # Objective mode
    ap.add_argument("--objective", default="target_alt", choices=["target_alt", "target_top1"],
                    help="Objective: target_alt = logit(target)-logit(alt); "
                         "target_top1 = logit(target)-logit(baseline_top1)")

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
    ap.add_argument("--include_negative", action="store_true",
                    help="Include neurons with pred <= 0 in ranking (default: positive only)")

    # Controls
    ap.add_argument("--controls", default="random,permute,negdelta,wrongpos",
                    help="Comma-separated controls: random, permute, permute_cross, negdelta, wrongpos, wrongsrc")
    ap.add_argument("--control_trials", type=int, default=5, help="Number of trials for stochastic controls")

    # Output
    ap.add_argument("--topk_tokens", type=int, default=10)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_plots", action="store_true", help="Skip plot generation")

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
    positive_only = not args.include_negative

    safe_print("=" * 70)
    safe_print("POC 3.5: Global Cross-Layer Steering at pos=-1")
    safe_print("=" * 70)
    safe_print(f"Model: {args.model_name} | dtype: {args.dtype} | seed: {args.seed}")
    safe_print(f"Layers: {layer_start}-{layer_end} ({len(layers)} layers)")
    safe_print(f"Dest: {args.prompt}")
    safe_print(f"Src:  {args.source_prompt}")
    safe_print(f"Target: {args.target_token!r} | Alt: {args.alt_token!r}")
    safe_print(f"Objective: {args.objective}")
    safe_print(f"Selection: {args.select_mode} | alpha: {args.alpha}")
    safe_print(f"Controls: {controls}")
    safe_print("")

    # Load model
    safe_print("Loading model...")
    model = load_model(args.model_name, args.device, dtype)
    model.eval()  # Ensure eval mode
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
    pos_arg = args.pos  # Original arg value (e.g., -1)
    pos_dest = normalize_pos(args.pos, dest_seq_len)  # Normalized (e.g., 8)
    pos_src = normalize_pos(args.pos, src_seq_len)

    safe_print(f"Dest seq_len: {dest_seq_len}, pos_arg: {pos_arg}, pos_normalized: {pos_dest}")
    safe_print(f"Src seq_len: {src_seq_len}, pos_arg: {pos_arg}, pos_normalized: {pos_src}")

    # Show token at pos
    dest_tok_at_pos = model.tokenizer.decode([dest_tokens[0, pos_dest].item()])
    src_tok_at_pos = model.tokenizer.decode([src_tokens[0, pos_src].item()])
    safe_print(f"Token at dest pos={pos_dest}: {sanitize_token(dest_tok_at_pos)!r}")
    safe_print(f"Token at src pos={pos_src}: {sanitize_token(src_tok_at_pos)!r}")
    safe_print("")

    # =========================================================================
    # Step 0: Determine objective token (for target_top1 mode)
    # =========================================================================
    objective_alt_id = alt_id
    objective_alt_token = args.alt_token

    if args.objective == "target_top1":
        safe_print("Step 0: Getting baseline top-1 for target_top1 objective...")
        baseline_top1_id = get_baseline_top1_id(model, dest_tokens)
        baseline_top1_token = sanitize_token(model.tokenizer.decode([baseline_top1_id]))
        safe_print(f"  Baseline top-1: {baseline_top1_token!r} (id={baseline_top1_id})")

        # Use baseline top-1 as the "alt" for gradient computation
        objective_alt_id = baseline_top1_id
        objective_alt_token = baseline_top1_token
        safe_print(f"  Objective: logit({args.target_token!r}) - logit({baseline_top1_token!r})")
        safe_print("")

    # =========================================================================
    # Step 1: Capture activations and gradients
    # =========================================================================
    safe_print("Step 1: Capturing activations and gradients (dest with grad, src no grad)...")

    acts_dest, grads, baseline_diff_obj = capture_acts_and_grads_dest(
        model, dest_tokens, layers, pos_dest, target_id, objective_alt_id
    )
    safe_print(f"  Baseline logit_diff (target-objective_alt): {baseline_diff_obj:+.4f}")

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
    # Step 3: Baseline metrics (dual: both target-alt AND target-objective_alt)
    # =========================================================================
    safe_print("\nStep 3: Computing baseline metrics (dual)...")

    with torch.inference_mode():
        base_logits = model(dest_tokens, return_type="logits")[0, -1]
    baseline_dual = compute_dual_metrics(model.tokenizer, base_logits, target_id, alt_id, objective_alt_id, args.topk_tokens)

    safe_print(f"  Baseline (target vs alt):")
    safe_print(f"    logit_diff={baseline_dual.target_alt.logit_diff:+.4f}")
    safe_print(f"    rank_target={baseline_dual.target_alt.rank_target}, rank_alt={baseline_dual.target_alt.rank_alt}")
    safe_print(f"    top1: {baseline_dual.target_alt.top1_token!r}")
    if not baseline_dual.same_objective:
        safe_print(f"  Baseline (target vs objective_alt):")
        safe_print(f"    logit_diff={baseline_dual.target_obj.logit_diff:+.4f}")

    # =========================================================================
    # Step 4: Selection and intervention
    # =========================================================================
    safe_print(f"\nStep 4: Running {args.select_mode} selection + intervention...")

    main_results = []
    selected_for_controls: List[NeuronCandidate] = []  # Track actual selection for controls

    if args.select_mode == "per_layer":
        selected = select_per_layer(candidates, args.k_per_layer)
        selected_for_controls = selected
        safe_print(f"  Selected {len(selected)} neurons ({args.k_per_layer} per layer)")

        plan = SteeringPlan(pos=pos_dest)
        for c in selected:
            plan.add(c)

        plan_by_layer = plan.by_layer()
        steered_logits = run_with_plan(model, dest_tokens, plan, args.alpha, plan_by_layer)
        steered_dual = compute_dual_metrics(model.tokenizer, steered_logits, target_id, alt_id, objective_alt_id, args.topk_tokens)

        d_logit_diff_alt = steered_dual.target_alt.logit_diff - baseline_dual.target_alt.logit_diff
        d_logit_diff_obj = steered_dual.target_obj.logit_diff - baseline_dual.target_obj.logit_diff
        d_target_logit = steered_dual.target_alt.logit_target - baseline_dual.target_alt.logit_target
        d_alt_logit = steered_dual.target_alt.logit_alt - baseline_dual.target_alt.logit_alt
        pred_sum = plan.total_pred * args.alpha
        eps = 1e-6

        main_results.append({
            "mode": "per_layer",
            "k": len(selected),
            "k_per_layer": args.k_per_layer,
            "pred_sum": pred_sum,
            "d_logit_diff": d_logit_diff_alt,
            "d_logit_diff_obj": d_logit_diff_obj,
            "d_target_logit": d_target_logit,
            "d_alt_logit": d_alt_logit,
            "pred_vs_actual": pred_sum / d_logit_diff_obj if abs(d_logit_diff_obj) > eps else float('nan'),
            "pred_minus_actual": pred_sum - d_logit_diff_obj,
            "steered": dual_metrics_to_dict(steered_dual),
            "success": steered_dual.target_alt.rank_target <= steered_dual.target_alt.rank_alt,
        })

    elif args.select_mode == "global":
        selected = select_global_topk(candidates, args.k_global)
        selected_for_controls = selected
        safe_print(f"  Selected {len(selected)} neurons (global top-{args.k_global})")

        plan = SteeringPlan(pos=pos_dest)
        for c in selected:
            plan.add(c)

        plan_by_layer = plan.by_layer()
        steered_logits = run_with_plan(model, dest_tokens, plan, args.alpha, plan_by_layer)
        steered_dual = compute_dual_metrics(model.tokenizer, steered_logits, target_id, alt_id, objective_alt_id, args.topk_tokens)

        d_logit_diff_alt = steered_dual.target_alt.logit_diff - baseline_dual.target_alt.logit_diff
        d_logit_diff_obj = steered_dual.target_obj.logit_diff - baseline_dual.target_obj.logit_diff
        d_target_logit = steered_dual.target_alt.logit_target - baseline_dual.target_alt.logit_target
        d_alt_logit = steered_dual.target_alt.logit_alt - baseline_dual.target_alt.logit_alt
        pred_sum = plan.total_pred * args.alpha
        eps = 1e-6

        main_results.append({
            "mode": "global",
            "k": len(selected),
            "pred_sum": pred_sum,
            "d_logit_diff": d_logit_diff_alt,
            "d_logit_diff_obj": d_logit_diff_obj,
            "d_target_logit": d_target_logit,
            "d_alt_logit": d_alt_logit,
            "pred_vs_actual": pred_sum / d_logit_diff_obj if abs(d_logit_diff_obj) > eps else float('nan'),
            "pred_minus_actual": pred_sum - d_logit_diff_obj,
            "steered": dual_metrics_to_dict(steered_dual),
            "success": steered_dual.target_alt.rank_target <= steered_dual.target_alt.rank_alt,
        })

    elif args.select_mode == "greedy":
        safe_print(f"  Running greedy batches: K = {k_steps}")
        main_results = run_greedy_batches(
            model, dest_tokens, candidates, pos_dest, args.alpha,
            k_steps, target_id, alt_id, objective_alt_id, baseline_dual, args.topk_tokens
        )

        safe_print("\n  Greedy results:")
        for r in main_results:
            status = "SUCCESS" if r["success"] else ""
            obj_note = f" (obj={r['d_logit_diff_obj']:+.4f})" if not baseline_dual.same_objective else ""
            safe_print(f"    K={r['k']:4d}: d_diff={r['d_logit_diff']:+.4f}{obj_note}, pred={r['pred_sum']:+.4f}, "
                      f"p/a={r['pred_vs_actual']:.2f}, layers={r['n_layers']} {status}")

        # For greedy mode: use best successful K, or largest K
        best_idx = len(main_results) - 1
        for i, r in enumerate(main_results):
            if r["success"]:
                best_idx = i
                break
        best_k = main_results[best_idx]["k"]
        selected_for_controls = candidates[:best_k]

    # Build the control plan from actual selection
    control_plan = SteeringPlan(pos=pos_dest)
    for c in selected_for_controls:
        control_plan.add(c)

    # Determine best_idx for non-greedy modes (they have only one result)
    if args.select_mode != "greedy":
        best_idx = 0

    # =========================================================================
    # Step 5: Controls
    # =========================================================================
    control_results = {}
    # IMPORTANT: Use best_idx consistently - controls are built on best_k, so compare to best_idx result
    main_d_diff = main_results[best_idx]["d_logit_diff"] if main_results else 0.0

    if "random" in controls:
        safe_print(f"\nControl: Random neurons (n={args.control_trials} trials)...")
        random_diffs = []

        for t in range(args.control_trials):
            rand_plan = make_random_plan_like(control_plan, layers, d_mlp)
            rand_logits = run_with_plan(model, dest_tokens, rand_plan, args.alpha)
            rand_m = compute_logit_metrics(model.tokenizer, rand_logits, target_id, alt_id, args.topk_tokens)
            random_diffs.append(rand_m.logit_diff - baseline_dual.target_alt.logit_diff)

        avg_rand = sum(random_diffs) / len(random_diffs)
        std_rand = (sum((x - avg_rand)**2 for x in random_diffs) / len(random_diffs)) ** 0.5
        control_results["random"] = {
            "d_logit_diff_mean": avg_rand,
            "d_logit_diff_std": std_rand,
            "samples": random_diffs,
            "specificity": abs(main_d_diff) / max(abs(avg_rand), 1e-6),
        }
        safe_print(f"  Random: d_diff_mean={avg_rand:+.4f} +/- {std_rand:.4f}, specificity={control_results['random']['specificity']:.1f}x")

    if "permute" in controls:
        safe_print(f"\nControl: Permuted deltas within-layer (n={args.control_trials} trials)...")
        perm_diffs = []

        for t in range(args.control_trials):
            perm_plan = make_permuted_plan_within_layer(control_plan)
            perm_logits = run_with_plan(model, dest_tokens, perm_plan, args.alpha)
            perm_m = compute_logit_metrics(model.tokenizer, perm_logits, target_id, alt_id, args.topk_tokens)
            perm_diffs.append(perm_m.logit_diff - baseline_dual.target_alt.logit_diff)

        avg_perm = sum(perm_diffs) / len(perm_diffs)
        std_perm = (sum((x - avg_perm)**2 for x in perm_diffs) / len(perm_diffs)) ** 0.5
        control_results["permute"] = {
            "d_logit_diff_mean": avg_perm,
            "d_logit_diff_std": std_perm,
            "samples": perm_diffs,
            "specificity": abs(main_d_diff) / max(abs(avg_perm), 1e-6),
        }
        safe_print(f"  Permute: d_diff_mean={avg_perm:+.4f} +/- {std_perm:.4f}, specificity={control_results['permute']['specificity']:.1f}x")

    if "permute_cross" in controls:
        safe_print(f"\nControl: Permuted deltas cross-layer (n={args.control_trials} trials)...")
        perm_cross_diffs = []

        for t in range(args.control_trials):
            perm_plan = make_permuted_plan_cross_layer(control_plan)
            perm_logits = run_with_plan(model, dest_tokens, perm_plan, args.alpha)
            perm_m = compute_logit_metrics(model.tokenizer, perm_logits, target_id, alt_id, args.topk_tokens)
            perm_cross_diffs.append(perm_m.logit_diff - baseline_dual.target_alt.logit_diff)

        avg_perm = sum(perm_cross_diffs) / len(perm_cross_diffs)
        std_perm = (sum((x - avg_perm)**2 for x in perm_cross_diffs) / len(perm_cross_diffs)) ** 0.5
        control_results["permute_cross"] = {
            "d_logit_diff_mean": avg_perm,
            "d_logit_diff_std": std_perm,
            "samples": perm_cross_diffs,
            "specificity": abs(main_d_diff) / max(abs(avg_perm), 1e-6),
        }
        safe_print(f"  Permute_cross: d_diff_mean={avg_perm:+.4f} +/- {std_perm:.4f}, specificity={control_results['permute_cross']['specificity']:.1f}x")

    if "negdelta" in controls:
        safe_print("\nControl: Negated deltas...")
        neg_plan = make_negdelta_plan(control_plan)
        neg_logits = run_with_plan(model, dest_tokens, neg_plan, args.alpha)
        neg_m = compute_logit_metrics(model.tokenizer, neg_logits, target_id, alt_id, args.topk_tokens)
        neg_d_diff = neg_m.logit_diff - baseline_dual.target_alt.logit_diff

        control_results["negdelta"] = {
            "d_logit_diff": neg_d_diff,
            "direction_correct": (main_d_diff > 0 and neg_d_diff < 0) or (main_d_diff < 0 and neg_d_diff > 0),
            "steered": metrics_to_dict(neg_m),
        }
        safe_print(f"  Negdelta: d_diff={neg_d_diff:+.4f} (direction_correct={control_results['negdelta']['direction_correct']})")

    if "wrongpos" in controls:
        # Wrong position control: apply same plan at pos-1 or pos-2
        safe_print("\nControl: Wrong position...")
        wrongpos_results = []

        for offset in [-1, -2]:
            wrong_pos = pos_dest + offset
            # Guard against out-of-bounds positions
            if wrong_pos < 0 or wrong_pos >= dest_seq_len:
                continue

            wrong_pos_plan = control_plan.with_new_pos(wrong_pos)
            wrong_pos_logits = run_with_plan(model, dest_tokens, wrong_pos_plan, args.alpha)
            wrong_pos_m = compute_logit_metrics(model.tokenizer, wrong_pos_logits, target_id, alt_id, args.topk_tokens)
            wrong_pos_d_diff = wrong_pos_m.logit_diff - baseline_dual.target_alt.logit_diff

            wrong_tok = model.tokenizer.decode([dest_tokens[0, wrong_pos].item()])
            wrongpos_results.append({
                "offset": offset,
                "pos": wrong_pos,
                "token_at_pos": sanitize_token(wrong_tok),
                "d_logit_diff": wrong_pos_d_diff,
            })
            safe_print(f"  pos={wrong_pos} ({sanitize_token(wrong_tok)!r}): d_diff={wrong_pos_d_diff:+.4f}")

        if wrongpos_results:
            avg_wrongpos = sum(r["d_logit_diff"] for r in wrongpos_results) / len(wrongpos_results)
            control_results["wrongpos"] = {
                "positions": wrongpos_results,
                "d_logit_diff_mean": avg_wrongpos,
                "specificity": abs(main_d_diff) / max(abs(avg_wrongpos), 1e-6),
            }
            safe_print(f"  WrongPos avg: d_diff_mean={avg_wrongpos:+.4f}, specificity={control_results['wrongpos']['specificity']:.1f}x")

    if "wrongsrc" in controls and args.wrong_source_prompt:
        safe_print("\nControl: Wrong source prompt...")
        wrong_tokens = model.to_tokens(args.wrong_source_prompt, prepend_bos=prepend_bos)
        pos_wrong = normalize_pos(args.pos, wrong_tokens.shape[1])
        acts_wrong = capture_acts_src(model, wrong_tokens, layers, pos_wrong)

        # Recompute deltas with wrong source
        wrong_candidates = rank_candidates_pos1(acts_dest, acts_wrong, grads, layers, positive_only=positive_only)
        wrong_plan = SteeringPlan(pos=pos_dest)
        for c in wrong_candidates[:len(selected_for_controls)]:
            wrong_plan.add(c)

        wrong_logits = run_with_plan(model, dest_tokens, wrong_plan, args.alpha)
        wrong_m = compute_logit_metrics(model.tokenizer, wrong_logits, target_id, alt_id, args.topk_tokens)
        wrong_d_diff = wrong_m.logit_diff - baseline_dual.target_alt.logit_diff

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

    safe_print(f"\nBaseline: logit_diff={baseline_dual.target_alt.logit_diff:+.4f}, rank_target={baseline_dual.target_alt.rank_target}, top1={baseline_dual.target_alt.top1_token!r}")

    if main_results:
        # Use best_idx consistently (same as controls were built on)
        best_r = main_results[best_idx]
        k_used = best_r.get('k', len(selected_for_controls))
        safe_print(f"\nBest intervention (K={k_used}):")
        safe_print(f"  d_logit_diff (target-alt) = {best_r['d_logit_diff']:+.4f}")
        if not baseline_dual.same_objective:
            safe_print(f"  d_logit_diff (objective)  = {best_r['d_logit_diff_obj']:+.4f}")
        safe_print(f"  d_target_logit            = {best_r['d_target_logit']:+.4f}")
        safe_print(f"  d_alt_logit               = {best_r['d_alt_logit']:+.4f}")
        safe_print(f"  pred/actual               = {best_r['pred_vs_actual']:.2f}")
        safe_print(f"  pred-actual               = {best_r['pred_minus_actual']:+.4f}")

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
        "objective": args.objective,
        "objective_alt_id": objective_alt_id,
        "objective_alt_token": objective_alt_token,
        "pos_arg": pos_arg,
        "pos_dest_normalized": pos_dest,
        "pos_src_normalized": pos_src,
        "dest_token_at_pos": sanitize_token(dest_tok_at_pos),
        "src_token_at_pos": sanitize_token(src_tok_at_pos),
        "layers": layers,
        "alpha": args.alpha,
        "select_mode": args.select_mode,
        "k_per_layer": args.k_per_layer,
        "k_global": args.k_global,
        "k_steps": k_steps,
        "positive_only": positive_only,
        "total_candidates": len(candidates),
        "baseline": dual_metrics_to_dict(baseline_dual),
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

    # Generate plots
    if not args.no_plots:
        safe_print("\nGenerating plots...")
        # Pass main_effect from best_idx (consistent with controls comparison)
        main_effect_for_plot = main_results[best_idx]["d_logit_diff"] if main_results else 0.0
        generate_plots(out_dir, main_results, control_results, baseline_dual.target_alt.logit_diff, 
                      main_effect_for_plot, best_idx)

    safe_print("\nDone.")


if __name__ == "__main__":
    main()
