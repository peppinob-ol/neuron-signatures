"""
POC 3.3: Advanced gradient-guided multi-neuron concept swap (teacher forcing, Gemma-2-2B).

Upgrades over POC 3.2:
1. Multi-position patching: patch at multiple token positions simultaneously
2. Stronger metrics: rank_target, margin_to_top1, probabilities, not just logit_diff
3. Additional controls:
   - Structured control: same neurons/grads, deltas from different entity (no extra grad passes)
   - Permuted-delta control: same neurons, shuffled deltas (destroys information)
   - Position-scramble control: apply deltas at wrong positions (tests position-specificity)
4. Per-position ablation: test each position individually
5. Auto layer selection: find best layer first, then run full sweeps
6. Better reporting: paper-grade JSON with all diagnostics

Run examples:

  # RECOMMENDED: Paper-style pos=-1 (last token, simple and robust)
  python -m neuron_signatures.poc3_3_grad_guided_swap_advanced \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --control_source_prompt "The capital of the state containing Miami is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer 19 --pos_mode last --pos -1 \\
    --enable_permute_delta_control --seed 42 --permute_trials 20

  # Single layer, single position (token mode)
  python -m neuron_signatures.poc3_3_grad_guided_swap_advanced \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer 19 --pos_mode token --pos_token_dest " Dallas" --pos_token_src " San"

  # Full controls (structured + permuted)
  python -m neuron_signatures.poc3_3_grad_guided_swap_advanced \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --control_source_prompt "The capital of the state containing Miami is" \\
    --enable_permute_delta_control \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer 19 --pos_mode token --pos_token_dest " Dallas" --pos_token_src " San" \\
    --seed 42 --permute_trials 20

  # Multi-position patching (Dallas + containing + is) with scramble control
  python -m neuron_signatures.poc3_3_grad_guided_swap_advanced \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --control_source_prompt "The capital of the state containing Miami is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer 19 --pos_mode multi \\
    --pos_multi_dest " Dallas| containing| is" \\
    --pos_multi_src " San| containing| is" \\
    --enable_permute_delta_control --scramble_trials 10

  # Auto-pick best layer, then sweep
  python -m neuron_signatures.poc3_3_grad_guided_swap_advanced \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --auto_pick_best_layer --layer_sweep 18,19,20,21,22 \\
    --pos_mode token --pos_token_dest " Dallas" --pos_token_src " San" \\
    --sweep_k 10,30,100 --sweep_alpha 0.5,1.0,2.0
"""

from __future__ import annotations

import argparse
import json
import random
import copy
from dataclasses import dataclass
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
class PosPlan:
    """Plan for patching at a single position."""
    dest_pos: int
    src_pos: int
    neuron_idxs: List[int]
    deltas: torch.Tensor       # [k] float32 CPU - (a_src - a_dest) for selected neurons
    grads: torch.Tensor        # [k] float32 CPU - gradients for selected neurons
    pred_terms: torch.Tensor   # [k] float32 CPU - delta*grad for selected neurons
    pred_sum: float
    a_dest_sel: torch.Tensor   # [k] float32 CPU - dest activations for selected neurons


@dataclass
class LogitMetrics:
    """Comprehensive logit-based metrics."""
    logit_target: float
    logit_alt: float
    logit_diff: float
    rank_target: int  # 0-indexed rank (0 = top-1)
    rank_alt: int
    margin_to_top1: float  # top1_logit - logit_target (0 if target is top-1)
    top1_token: str
    top1_logit: float
    target_prob: float
    alt_prob: float
    topk: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Logit helpers
# -----------------------------------------------------------------------------

def compute_logit_metrics(
    tokenizer,
    logits_1d: torch.Tensor,
    target_id: int,
    alt_id: int,
    topk_n: int = 10,
) -> LogitMetrics:
    """Compute comprehensive metrics from logits (float32)."""
    logits = logits_1d.float().cpu()
    
    lt = float(logits[target_id].item())
    la = float(logits[alt_id].item())
    
    # Ranks (0-indexed)
    rank_target = int((logits > lt).sum().item())
    rank_alt = int((logits > la).sum().item())
    
    # Top-1
    top1_idx = int(logits.argmax().item())
    top1_logit = float(logits[top1_idx].item())
    top1_token = sanitize_token(tokenizer.decode([top1_idx]))
    
    # Margin to top-1
    margin = top1_logit - lt if top1_idx != target_id else 0.0
    
    # Probabilities (full softmax in float32)
    probs = torch.softmax(logits, dim=-1)
    target_prob = float(probs[target_id].item())
    alt_prob = float(probs[alt_id].item())
    
    # Top-k
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
        margin_to_top1=margin,
        top1_token=top1_token,
        top1_logit=top1_logit,
        target_prob=target_prob,
        alt_prob=alt_prob,
        topk=topk,
    )


def metrics_to_dict(m: LogitMetrics) -> Dict[str, Any]:
    return {
        "logit_target": m.logit_target,
        "logit_alt": m.logit_alt,
        "logit_diff": m.logit_diff,
        "rank_target": m.rank_target,
        "rank_alt": m.rank_alt,
        "margin_to_top1": m.margin_to_top1,
        "top1_token": m.top1_token,
        "top1_logit": m.top1_logit,
        "target_prob": m.target_prob,
        "alt_prob": m.alt_prob,
        "topk": m.topk,
    }


# -----------------------------------------------------------------------------
# Forward pass helpers
# -----------------------------------------------------------------------------

def forward_with_cache(
    model,
    prompt: str,
    layer: int,
    prepend_bos: bool = True,
) -> Tuple[torch.Tensor, Any, torch.Tensor]:
    """Returns: logits [1, seq, vocab], cache, tokens [1, seq_len]"""
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    key = f"blocks.{layer}.mlp.hook_post"

    def names_filter(name: str) -> bool:
        return name == key

    with torch.inference_mode():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter, return_type="logits")

    return logits, cache, tokens


def find_token_position(model, tokens: torch.Tensor, token_str: str) -> int:
    """Find position of token string in sequence (last occurrence)."""
    token_id = resolve_single_token_id(model, token_str)
    token_list = tokens[0].tolist()
    
    for i in range(len(token_list) - 1, -1, -1):
        if token_list[i] == token_id:
            return i
    
    # Partial match fallback
    for i in range(len(token_list) - 1, -1, -1):
        decoded = model.tokenizer.decode([token_list[i]])
        if token_str.strip() in decoded or decoded in token_str:
            return i
    
    raise ValueError(f"Token {token_str!r} not found in sequence.")


def find_multi_positions(
    model,
    tokens: torch.Tensor,
    token_strs: List[str],
) -> List[int]:
    """Find positions for multiple tokens."""
    positions = []
    for ts in token_strs:
        positions.append(find_token_position(model, tokens, ts))
    return positions


def get_token_context(model, tokens: torch.Tensor, pos: int, window: int = 2) -> Dict[str, Any]:
    """Get token context around a position for debugging."""
    seq_len = tokens.shape[1]
    token_list = tokens[0].tolist()
    
    start = max(0, pos - window)
    end = min(seq_len, pos + window + 1)
    
    context_tokens = []
    for i in range(start, end):
        tok_str = sanitize_token(model.tokenizer.decode([token_list[i]]))
        context_tokens.append({
            "pos": i,
            "token": tok_str,
            "id": token_list[i],
            "is_target": i == pos,
        })
    
    return {
        "target_pos": pos,
        "target_token": sanitize_token(model.tokenizer.decode([token_list[pos]])),
        "target_id": token_list[pos],
        "context": context_tokens,
    }


# -----------------------------------------------------------------------------
# Gradient computation (supports multiple positions)
# -----------------------------------------------------------------------------

def compute_grads_multi_pos(
    model,
    tokens: torch.Tensor,
    layer: int,
    positions: List[int],
    target_id: int,
    alt_id: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
    """
    Compute gradients at multiple positions.
    
    Returns:
        acts: List of [d_mlp] float32 CPU tensors (activations at each pos)
        grads: List of [d_mlp] float32 CPU tensors (gradients at each pos)
        baseline_diff: float - the logit_diff from this forward pass
    """
    hook_name = f"blocks.{layer}.mlp.hook_post"
    captured: Dict[str, torch.Tensor] = {}

    def capture_with_grad(act: torch.Tensor, hook) -> torch.Tensor:
        act.retain_grad()
        captured["act"] = act
        return act

    model.reset_hooks(including_permanent=True)
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, capture_with_grad)],
            return_type="logits",
        )
        lt = logits[0, -1, target_id].float()
        la = logits[0, -1, alt_id].float()
        diff = lt - la
        diff.backward()

    model.reset_hooks(including_permanent=True)

    if "act" not in captured:
        raise RuntimeError(f"Hook did not capture activation for {hook_name}")
    
    act_full = captured["act"]
    grad_full = act_full.grad

    if grad_full is None:
        raise RuntimeError("Gradient is None")

    acts = [act_full[0, p].detach().float().cpu() for p in positions]
    grads = [grad_full[0, p].detach().float().cpu() for p in positions]
    baseline_diff = float(diff.detach().item())

    return acts, grads, baseline_diff


# -----------------------------------------------------------------------------
# Neuron selection
# -----------------------------------------------------------------------------

def select_neurons_for_pos(
    a_dest: torch.Tensor,
    a_src: torch.Tensor,
    grad: torch.Tensor,
    k: int,
    selection_mode: str = "pos",
) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select neurons for one position.
    
    Returns:
        neuron_idxs: list of selected indices
        deltas: [k] deltas for selected neurons
        grads_sel: [k] gradients for selected neurons
        pred_terms: [k] delta*grad for selected neurons
        a_dest_sel: [k] dest activations for selected neurons
    """
    delta_a = (a_src - a_dest).float()
    pred = (delta_a * grad).detach()
    
    if selection_mode == "pos":
        pos_mask = pred > 0
        pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
        
        if pos_idx.numel() == 0:
            return [], torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        
        vals = pred[pos_idx]
        k_eff = min(k, vals.numel())
        top_result = torch.topk(vals, k_eff)
        chosen = pos_idx[top_result.indices]
        
    elif selection_mode == "abs":
        k_eff = min(k, pred.numel())
        top_result = torch.topk(pred.abs(), k_eff)
        chosen = top_result.indices
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
    
    neuron_idxs = chosen.tolist()
    deltas = delta_a[chosen]
    grads_sel = grad[chosen]
    pred_terms = pred[chosen]
    a_dest_sel = a_dest[chosen].clone()
    
    return neuron_idxs, deltas, grads_sel, pred_terms, a_dest_sel


def build_pos_plans(
    model,
    dest_tokens: torch.Tensor,
    src_cache,
    layer: int,
    dest_positions: List[int],
    src_positions: List[int],
    target_id: int,
    alt_id: int,
    k_per_pos: int,
    selection_mode: str,
) -> Tuple[List[PosPlan], float, List[torch.Tensor]]:
    """
    Build patching plans for multiple positions.
    
    Returns:
        plans: List of PosPlan
        baseline_diff: float
        acts_dest_full: List of full [d_mlp] dest activation vectors (for controls)
    """
    # Compute gradients at all dest positions
    acts_dest, grads, baseline_diff = compute_grads_multi_pos(
        model, dest_tokens, layer, dest_positions, target_id, alt_id
    )
    
    hook_key = f"blocks.{layer}.mlp.hook_post"
    
    plans = []
    for i, (dest_pos, src_pos) in enumerate(zip(dest_positions, src_positions)):
        a_dest = acts_dest[i]
        grad = grads[i]
        a_src = src_cache[hook_key][0, src_pos].detach().float().cpu()
        
        neuron_idxs, deltas, grads_sel, pred_terms, a_dest_sel = select_neurons_for_pos(
            a_dest, a_src, grad, k_per_pos, selection_mode
        )
        
        if len(neuron_idxs) == 0:
            safe_print(f"  WARNING: pos={dest_pos} selected 0 neurons (no positive pred terms)")
            continue
        
        plan = PosPlan(
            dest_pos=dest_pos,
            src_pos=src_pos,
            neuron_idxs=neuron_idxs,
            deltas=deltas,
            grads=grads_sel,
            pred_terms=pred_terms,
            pred_sum=float(pred_terms.sum().item()),
            a_dest_sel=a_dest_sel,
        )
        plans.append(plan)
    
    return plans, baseline_diff, acts_dest


def build_control_plans_same_neurons(
    plans: List[PosPlan],
    ctrl_cache,
    ctrl_positions: List[int],
    hook_key: str,
) -> List[PosPlan]:
    """
    Build control plans using SAME neurons and grads, only change deltas.
    
    This is the correct way to do structured control:
    - Same neuron_idxs (from main selection)
    - Same grads (from dest prompt gradient)
    - New deltas: a_ctrl[idx] - a_dest[idx]
    
    NO gradient recomputation needed.
    """
    ctrl_plans = []
    for plan, ctrl_pos in zip(plans, ctrl_positions):
        idx_t = torch.tensor(plan.neuron_idxs, dtype=torch.long)
        a_ctrl = ctrl_cache[hook_key][0, ctrl_pos].detach().float().cpu()
        
        # Control deltas: what we'd add if patching from ctrl instead of src
        ctrl_deltas = a_ctrl[idx_t] - plan.a_dest_sel
        ctrl_pred_terms = ctrl_deltas * plan.grads
        
        ctrl_plan = PosPlan(
            dest_pos=plan.dest_pos,
            src_pos=ctrl_pos,
            neuron_idxs=plan.neuron_idxs,
            deltas=ctrl_deltas,
            grads=plan.grads,
            pred_terms=ctrl_pred_terms,
            pred_sum=float(ctrl_pred_terms.sum().item()),
            a_dest_sel=plan.a_dest_sel,  # Keep original for reference
        )
        ctrl_plans.append(ctrl_plan)
    
    return ctrl_plans


# -----------------------------------------------------------------------------
# Intervention hooks
# -----------------------------------------------------------------------------

def make_hook_patch_delta_multi(
    layer: int,
    plans: List[PosPlan],
    alpha: float,
):
    """Hook that applies patches at multiple positions."""
    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        act2 = act.clone()
        for plan in plans:
            idx = torch.tensor(plan.neuron_idxs, device=act2.device, dtype=torch.long)
            d = (alpha * plan.deltas).to(device=act2.device, dtype=act2.dtype)
            act2[:, plan.dest_pos, idx] = act2[:, plan.dest_pos, idx] + d
        return act2
    return hook_fn


def make_hook_patch_delta_permuted(
    layer: int,
    plans: List[PosPlan],
    alpha: float,
):
    """Hook with permuted deltas (control: destroys information)."""
    # Pre-compute permutations for reproducibility within the hook
    permuted_deltas = []
    for plan in plans:
        perm = torch.randperm(len(plan.neuron_idxs))
        permuted_deltas.append(plan.deltas[perm])
    
    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        act2 = act.clone()
        for plan, p_deltas in zip(plans, permuted_deltas):
            idx = torch.tensor(plan.neuron_idxs, device=act2.device, dtype=torch.long)
            d = (alpha * p_deltas).to(device=act2.device, dtype=act2.dtype)
            act2[:, plan.dest_pos, idx] = act2[:, plan.dest_pos, idx] + d
        return act2
    return hook_fn


def make_hook_patch_delta_scrambled_positions(
    plans: List[PosPlan],
    alpha: float,
    scrambled_dest_positions: List[int],
):
    """
    Hook that applies deltas at scrambled positions (control: tests position-specificity).
    
    Each plan's own (neuron_idxs, deltas) is applied at a different dest_pos.
    This tests whether the effect is position-specific or just "injecting energy anywhere".
    
    Args:
        plans: Original plans with neuron_idxs and deltas
        alpha: Scaling factor
        scrambled_dest_positions: Permuted list of dest positions (same length as plans)
    """
    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        act2 = act.clone()
        for plan, new_pos in zip(plans, scrambled_dest_positions):
            idx = torch.tensor(plan.neuron_idxs, device=act2.device, dtype=torch.long)
            d = (alpha * plan.deltas).to(device=act2.device, dtype=act2.dtype)
            act2[:, new_pos, idx] = act2[:, new_pos, idx] + d
        return act2
    return hook_fn


def make_hook_patch_delta_shuffled_neurons(
    plans: List[PosPlan],
    alpha: float,
):
    """
    Hook that applies deltas to WRONG neurons at the SAME position (strongest control).
    
    For each plan, we shuffle the neuron indices while keeping deltas fixed.
    This breaks the delta<->neuron correspondence: we're injecting the same "energy"
    but at wrong neurons. If the effect collapses, it proves the intervention is
    neuron-specific, not just perturbation.
    
    Args:
        plans: Original plans with neuron_idxs and deltas
        alpha: Scaling factor
    """
    # Pre-compute shuffled indices for each plan
    shuffled_indices = []
    for plan in plans:
        k = len(plan.neuron_idxs)
        perm = torch.randperm(k)
        # Shuffle the neuron indices, keep deltas in original order
        # So delta[i] goes to neuron_idxs[perm[i]] instead of neuron_idxs[i]
        shuffled_idx = [plan.neuron_idxs[perm[j].item()] for j in range(k)]
        shuffled_indices.append(torch.tensor(shuffled_idx, dtype=torch.long))
    
    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        act2 = act.clone()
        for plan, shuf_idx in zip(plans, shuffled_indices):
            idx = shuf_idx.to(device=act2.device)
            d = (alpha * plan.deltas).to(device=act2.device, dtype=act2.dtype)
            act2[:, plan.dest_pos, idx] = act2[:, plan.dest_pos, idx] + d
        return act2
    return hook_fn


def run_intervention(
    model,
    tokens: torch.Tensor,
    layer: int,
    hook_fn,
) -> torch.Tensor:
    """Run forward with hook, return next-token logits [vocab]."""
    hook_name = f"blocks.{layer}.mlp.hook_post"
    with torch.inference_mode():
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, hook_fn)],
            return_type="logits",
        )
    return logits[0, -1]


# -----------------------------------------------------------------------------
# Main experiment runner
# -----------------------------------------------------------------------------

def run_experiment(
    model,
    dest_tokens: torch.Tensor,
    src_cache,
    ctrl_cache,  # structured control cache (optional)
    layer: int,
    dest_positions: List[int],
    src_positions: List[int],
    ctrl_positions: Optional[List[int]],  # control positions
    target_id: int,
    alt_id: int,
    k_per_pos: int,
    alpha: float,
    selection_mode: str,
    enable_permute_control: bool,
    permute_trials: int,
    scramble_trials: int,
    enable_shuffle_neuron_control: bool,
    shuffle_neuron_trials: int,
    topk_n: int,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    
    hook_key = f"blocks.{layer}.mlp.hook_post"
    
    # Build plans using main source
    plans, baseline_diff, acts_dest_full = build_pos_plans(
        model, dest_tokens, src_cache, layer,
        dest_positions, src_positions,
        target_id, alt_id, k_per_pos, selection_mode
    )
    
    if len(plans) == 0:
        return {
            "success": False,
            "error": "No neurons selected (all positions empty)",
            "layer": layer,
            "requested_positions": len(dest_positions),
            "built_plans": 0,
        }
    
    if len(plans) < len(dest_positions):
        safe_print(f"  WARNING: Only {len(plans)}/{len(dest_positions)} positions have neurons")
    
    # Compute predicted effect
    total_pred = alpha * sum(p.pred_sum for p in plans)
    total_k = sum(len(p.neuron_idxs) for p in plans)
    
    # Get baseline metrics (explicit return_type for safety)
    with torch.inference_mode():
        base_logits_full = model(dest_tokens, return_type="logits")
        base_logits = base_logits_full[0, -1]
    baseline = compute_logit_metrics(model.tokenizer, base_logits, target_id, alt_id, topk_n)
    
    # Run main intervention
    hook_main = make_hook_patch_delta_multi(layer, plans, alpha)
    steered_logits = run_intervention(model, dest_tokens, layer, hook_main)
    steered = compute_logit_metrics(model.tokenizer, steered_logits, target_id, alt_id, topk_n)
    
    # Compute deltas
    d_logit_diff = steered.logit_diff - baseline.logit_diff
    d_margin = steered.margin_to_top1 - baseline.margin_to_top1
    d_rank = steered.rank_target - baseline.rank_target
    
    # Compute pred/actual carefully
    if abs(d_logit_diff) > 1e-6:
        pred_vs_actual = total_pred / d_logit_diff
        abs_error = total_pred - d_logit_diff
    else:
        pred_vs_actual = float('nan')
        abs_error = float('nan')
    
    result = {
        "success": True,
        "layer": layer,
        "n_positions": len(plans),
        "requested_positions": len(dest_positions),
        "total_neurons": total_k,
        "k_per_pos": k_per_pos,
        "alpha": alpha,
        "selection_mode": selection_mode,
        "pred_effect": total_pred,
        "baseline": metrics_to_dict(baseline),
        "steered": metrics_to_dict(steered),
        "d_logit_diff": d_logit_diff,
        "d_margin_to_top1": d_margin,
        "d_rank_target": d_rank,
        "pred_vs_actual": pred_vs_actual,
        "abs_error": abs_error,
        "plans": [
            {
                "dest_pos": p.dest_pos,
                "src_pos": p.src_pos,
                "n_neurons": len(p.neuron_idxs),
                "pred_sum": p.pred_sum,
                "top5_neurons": [
                    {"idx": int(p.neuron_idxs[i]), 
                     "delta": float(p.deltas[i].item()),
                     "grad": float(p.grads[i].item()),
                     "term": float(p.pred_terms[i].item())}
                    for i in range(min(5, len(p.neuron_idxs)))
                ]
            }
            for p in plans
        ],
    }
    
    # -------------------------------------------------------------------------
    # Per-position ablation (run each position alone)
    # -------------------------------------------------------------------------
    if len(plans) > 1:
        per_pos_effects = []
        for i, plan in enumerate(plans):
            hook_single = make_hook_patch_delta_multi(layer, [plan], alpha)
            single_logits = run_intervention(model, dest_tokens, layer, hook_single)
            single_metrics = compute_logit_metrics(model.tokenizer, single_logits, target_id, alt_id, topk_n)
            single_d_diff = single_metrics.logit_diff - baseline.logit_diff
            per_pos_effects.append({
                "dest_pos": plan.dest_pos,
                "d_logit_diff": single_d_diff,
                "pred_sum": plan.pred_sum * alpha,
                "n_neurons": len(plan.neuron_idxs),
            })
        result["per_position_effects"] = per_pos_effects
    
    # -------------------------------------------------------------------------
    # Control A: Structured control (same neurons/grads, different entity deltas)
    # NO gradient recomputation - uses exact same neurons and grads from main plans
    # -------------------------------------------------------------------------
    if ctrl_cache is not None and ctrl_positions is not None:
        ctrl_plans = build_control_plans_same_neurons(
            plans, ctrl_cache, ctrl_positions, hook_key
        )
        
        # Sanity asserts: ensure we're using SAME neurons/grads
        assert len(ctrl_plans) == len(plans), f"ctrl_plans length mismatch: {len(ctrl_plans)} vs {len(plans)}"
        for p, cp in zip(plans, ctrl_plans):
            assert p.neuron_idxs == cp.neuron_idxs, "Control plan has different neuron_idxs!"
        
        ctrl_pred = alpha * sum(p.pred_sum for p in ctrl_plans)
        hook_ctrl = make_hook_patch_delta_multi(layer, ctrl_plans, alpha)
        ctrl_logits = run_intervention(model, dest_tokens, layer, hook_ctrl)
        ctrl_metrics = compute_logit_metrics(model.tokenizer, ctrl_logits, target_id, alt_id, topk_n)
        ctrl_d_diff = ctrl_metrics.logit_diff - baseline.logit_diff
        
        ctrl_pva = ctrl_pred / ctrl_d_diff if abs(ctrl_d_diff) > 1e-6 else float('nan')
        
        result["control_structured"] = {
            "d_logit_diff": ctrl_d_diff,
            "pred_effect": ctrl_pred,
            "pred_vs_actual": ctrl_pva,
            "specificity_ratio": abs(d_logit_diff) / max(abs(ctrl_d_diff), 1e-6),
        }
    
    # -------------------------------------------------------------------------
    # Control B: Permuted deltas (same neurons, shuffled deltas within each pos)
    # Each trial uses a deterministic seed for reproducibility
    # -------------------------------------------------------------------------
    if enable_permute_control and len(plans) > 0 and permute_trials > 0:
        perm_d_diffs = []
        base_seed = torch.initial_seed()  # Capture current seed state
        for t in range(permute_trials):
            # Deterministic per-trial seed for reproducibility
            torch.manual_seed(base_seed + 10_000 + t)
            hook_perm = make_hook_patch_delta_permuted(layer, plans, alpha)
            perm_logits = run_intervention(model, dest_tokens, layer, hook_perm)
            perm_metrics = compute_logit_metrics(model.tokenizer, perm_logits, target_id, alt_id, topk_n)
            perm_d_diffs.append(perm_metrics.logit_diff - baseline.logit_diff)
        # Restore seed
        torch.manual_seed(base_seed)
        
        avg_perm = sum(perm_d_diffs) / len(perm_d_diffs)
        std_perm = (sum((x - avg_perm)**2 for x in perm_d_diffs) / len(perm_d_diffs)) ** 0.5
        
        result["control_permuted"] = {
            "d_logit_diff_mean": avg_perm,
            "d_logit_diff_std": std_perm,
            "d_logit_diff_samples": perm_d_diffs,
            "n_trials": permute_trials,
            "specificity_ratio": abs(d_logit_diff) / max(abs(avg_perm), 1e-6),
        }
    
    # -------------------------------------------------------------------------
    # Control C: Position-scramble (multi-pos only) - apply deltas at wrong positions
    # Each plan's (neuron_idxs, deltas) is applied at a different dest_pos
    # -------------------------------------------------------------------------
    if len(plans) > 1 and scramble_trials > 0:
        scramble_d_diffs = []
        orig_positions = [p.dest_pos for p in plans]
        base_seed = torch.initial_seed()
        
        for t in range(scramble_trials):
            # Deterministic per-trial seed
            random.seed(base_seed + 20_000 + t)
            
            # Permute the dest positions (not plan indices)
            new_positions = orig_positions.copy()
            random.shuffle(new_positions)
            # Ensure at least one position changed
            if new_positions == orig_positions and len(new_positions) > 1:
                new_positions[0], new_positions[1] = new_positions[1], new_positions[0]
            
            hook_scramble = make_hook_patch_delta_scrambled_positions(plans, alpha, new_positions)
            scramble_logits = run_intervention(model, dest_tokens, layer, hook_scramble)
            scramble_metrics = compute_logit_metrics(model.tokenizer, scramble_logits, target_id, alt_id, topk_n)
            scramble_d_diffs.append(scramble_metrics.logit_diff - baseline.logit_diff)
        
        # Restore seed
        random.seed(base_seed)
        
        avg_scramble = sum(scramble_d_diffs) / len(scramble_d_diffs)
        std_scramble = (sum((x - avg_scramble)**2 for x in scramble_d_diffs) / len(scramble_d_diffs)) ** 0.5
        
        result["control_scrambled_pos"] = {
            "d_logit_diff_mean": avg_scramble,
            "d_logit_diff_std": std_scramble,
            "d_logit_diff_samples": scramble_d_diffs,
            "n_trials": scramble_trials,
            "specificity_ratio": abs(d_logit_diff) / max(abs(avg_scramble), 1e-6),
        }
    
    # -------------------------------------------------------------------------
    # Control D: Shuffle-neuron (strongest control) - apply deltas to WRONG neurons
    # Breaks delta<->neuron correspondence: same energy, wrong target neurons
    # -------------------------------------------------------------------------
    if enable_shuffle_neuron_control and len(plans) > 0 and shuffle_neuron_trials > 0:
        shuffle_d_diffs = []
        base_seed = torch.initial_seed()
        
        for t in range(shuffle_neuron_trials):
            # Deterministic per-trial seed
            torch.manual_seed(base_seed + 30_000 + t)
            hook_shuffle = make_hook_patch_delta_shuffled_neurons(plans, alpha)
            shuffle_logits = run_intervention(model, dest_tokens, layer, hook_shuffle)
            shuffle_metrics = compute_logit_metrics(model.tokenizer, shuffle_logits, target_id, alt_id, topk_n)
            shuffle_d_diffs.append(shuffle_metrics.logit_diff - baseline.logit_diff)
        
        # Restore seed
        torch.manual_seed(base_seed)
        
        avg_shuffle = sum(shuffle_d_diffs) / len(shuffle_d_diffs)
        std_shuffle = (sum((x - avg_shuffle)**2 for x in shuffle_d_diffs) / len(shuffle_d_diffs)) ** 0.5
        
        result["control_shuffled_neurons"] = {
            "d_logit_diff_mean": avg_shuffle,
            "d_logit_diff_std": std_shuffle,
            "d_logit_diff_samples": shuffle_d_diffs,
            "n_trials": shuffle_neuron_trials,
            "specificity_ratio": abs(d_logit_diff) / max(abs(avg_shuffle), 1e-6),
        }
    
    return result


def auto_pick_best_layer(
    model,
    prompt: str,
    source_prompt: str,
    layers: List[int],
    dest_positions: List[int],
    src_positions: List[int],
    target_id: int,
    alt_id: int,
    k_per_pos: int,
    alpha: float,
    selection_mode: str,
    prepend_bos: bool,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Quick layer sweep to find best layer.
    
    Returns:
        best_layer: int
        sweep_results: list of dicts with layer and d_logit_diff
    """
    dest_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    
    results = []
    safe_print("Auto-picking best layer...")
    for layer in tqdm(layers, desc="Layer sweep", ascii=True):
        _, src_cache, src_tokens = forward_with_cache(model, source_prompt, layer, prepend_bos)
        
        result = run_experiment(
            model, dest_tokens, src_cache, None,
            layer, dest_positions, src_positions, None,
            target_id, alt_id, k_per_pos, alpha, selection_mode,
            enable_permute_control=False, permute_trials=0, scramble_trials=0,
            enable_shuffle_neuron_control=False, shuffle_neuron_trials=0, topk_n=5
        )
        
        if result["success"]:
            results.append({
                "layer": layer,
                "d_logit_diff": result["d_logit_diff"],
                "pred_effect": result["pred_effect"],
            })
            safe_print(f"  L{layer}: d_diff={result['d_logit_diff']:+.4f}")
        else:
            results.append({"layer": layer, "d_logit_diff": 0.0, "error": result.get("error")})
    
    if not results:
        raise ValueError("No successful layer runs")
    
    best = max(results, key=lambda x: x.get("d_logit_diff", float('-inf')))
    safe_print(f"Best layer: {best['layer']} with d_diff={best['d_logit_diff']:+.4f}")
    
    return best["layer"], results


def main():
    ap = argparse.ArgumentParser(description="POC 3.3: Advanced gradient-guided concept swap")
    
    # Model
    ap.add_argument("--model_name", default="google/gemma-2-2b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--no_prepend_bos", action="store_true")
    
    # Prompts
    ap.add_argument("--prompt", required=True, help="Destination prompt")
    ap.add_argument("--source_prompt", required=True, help="Main source prompt")
    ap.add_argument("--control_source_prompt", default=None, help="Structured control (same template, different entity)")
    ap.add_argument("--enable_permute_delta_control", action="store_true", help="Enable permuted-delta control")
    
    # Tokens
    ap.add_argument("--target_token", required=True)
    ap.add_argument("--alt_token", required=True)
    
    # Layer
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--layer_sweep", default=None, help="Comma-separated layers")
    ap.add_argument("--auto_pick_best_layer", action="store_true", help="Auto-pick best layer first")
    
    # Position
    ap.add_argument("--pos_mode", default="last", choices=["last", "token", "multi"])
    ap.add_argument("--pos", type=int, default=-1)
    ap.add_argument("--pos_token_dest", default=None)
    ap.add_argument("--pos_token_src", default=None)
    ap.add_argument("--pos_token_ctrl", default=None, 
                    help="Token to locate position in control_source_prompt (token mode). If omitted, tries pos_token_src then falls back.")
    ap.add_argument("--pos_multi_dest", default=None, help="Pipe-separated tokens for multi-pos")
    ap.add_argument("--pos_multi_src", default=None, help="Pipe-separated tokens for multi-pos")
    ap.add_argument("--ctrl_pos_policy", default="strict_token", choices=["strict_token", "same_index"],
                    help="How to choose ctrl position. strict_token=use pos_token_ctrl/pos_token_src; same_index=use src_positions index")
    
    # Intervention
    ap.add_argument("--k_per_pos", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--selection_mode", default="pos", choices=["pos", "abs"])
    
    # Sweeps
    ap.add_argument("--sweep_k", default=None)
    ap.add_argument("--sweep_alpha", default=None)
    
    # Controls
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    ap.add_argument("--permute_trials", type=int, default=10, help="Number of permuted-delta trials")
    ap.add_argument("--scramble_trials", type=int, default=5, help="Number of position-scramble trials (multi-pos only)")
    ap.add_argument("--enable_shuffle_neuron_control", action="store_true", 
                    help="Enable shuffle-neuron control (apply deltas to wrong neurons at same position)")
    ap.add_argument("--shuffle_neuron_trials", type=int, default=10, help="Number of shuffle-neuron trials")
    
    # Output
    ap.add_argument("--topk_tokens", type=int, default=10)
    ap.add_argument("--out_dir", default=None)
    
    args = ap.parse_args()
    
    # Validate
    if args.layer is None and args.layer_sweep is None and not args.auto_pick_best_layer:
        raise ValueError("Must specify --layer, --layer_sweep, or --auto_pick_best_layer")
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    dtype = DTYPE_MAP[args.dtype]
    prepend_bos = not args.no_prepend_bos
    
    safe_print("=== POC 3.3: Advanced Gradient-Guided Swap ===")
    safe_print(f"Model: {args.model_name} | dtype: {args.dtype} | seed: {args.seed}")
    safe_print(f"Dest: {args.prompt}")
    safe_print(f"Src: {args.source_prompt}")
    if args.control_source_prompt:
        safe_print(f"Ctrl: {args.control_source_prompt}")
    safe_print("")
    
    # Load model
    model = load_model(args.model_name, args.device, dtype)
    target_id = resolve_single_token_id(model, args.target_token)
    alt_id = resolve_single_token_id(model, args.alt_token)
    
    safe_print(f"Target: {args.target_token!r} (id={target_id})")
    safe_print(f"Alt: {args.alt_token!r} (id={alt_id})")
    safe_print("")
    
    # Tokenize prompts
    dest_tokens = model.to_tokens(args.prompt, prepend_bos=prepend_bos)
    src_tokens = model.to_tokens(args.source_prompt, prepend_bos=prepend_bos)
    ctrl_tokens = None
    if args.control_source_prompt:
        ctrl_tokens = model.to_tokens(args.control_source_prompt, prepend_bos=prepend_bos)
    
    # Resolve positions
    dest_seq_len = dest_tokens.shape[1]
    src_seq_len = src_tokens.shape[1]
    
    if args.pos_mode == "last":
        dest_positions = [normalize_pos(args.pos, dest_seq_len)]
        src_positions = [normalize_pos(args.pos, src_seq_len)]
    elif args.pos_mode == "token":
        if not args.pos_token_dest or not args.pos_token_src:
            raise ValueError("pos_mode=token requires --pos_token_dest and --pos_token_src")
        dest_positions = [find_token_position(model, dest_tokens, args.pos_token_dest)]
        src_positions = [find_token_position(model, src_tokens, args.pos_token_src)]
    elif args.pos_mode == "multi":
        if not args.pos_multi_dest or not args.pos_multi_src:
            raise ValueError("pos_mode=multi requires --pos_multi_dest and --pos_multi_src")
        dest_strs = [s.strip() for s in args.pos_multi_dest.split("|")]
        src_strs = [s.strip() for s in args.pos_multi_src.split("|")]
        if len(dest_strs) != len(src_strs):
            raise ValueError("pos_multi_dest and pos_multi_src must have same number of tokens")
        dest_positions = find_multi_positions(model, dest_tokens, dest_strs)
        src_positions = find_multi_positions(model, src_tokens, src_strs)
    
    # Resolve control positions (paper-grade: no silent fallbacks, explicit policies)
    ctrl_positions = None
    if ctrl_tokens is not None:
        if args.pos_mode == "last":
            # Last position mode: simple and robust
            ctrl_positions = [normalize_pos(args.pos, ctrl_tokens.shape[1])]
        
        elif args.pos_mode == "token":
            if args.ctrl_pos_policy == "same_index":
                # Use the same absolute index as src_positions (template-slot alignment)
                # Only valid if control prompt tokenization is compatible
                sp = src_positions[0]
                if sp < 0 or sp >= ctrl_tokens.shape[1]:
                    safe_print(f"WARNING: src_pos={sp} out of range for ctrl prompt (len={ctrl_tokens.shape[1]}); structured control disabled.")
                    ctrl_tokens = None
                else:
                    ctrl_positions = [sp]
            else:
                # strict_token: find explicit token in ctrl prompt
                ctrl_tok = args.pos_token_ctrl or args.pos_token_src
                try:
                    ctrl_positions = [find_token_position(model, ctrl_tokens, ctrl_tok)]
                except ValueError as e:
                    # IMPORTANT: token-mode structured control must be position-matched
                    safe_print(f"WARNING: ctrl token {ctrl_tok!r} not found in control prompt; structured control disabled. ({e})")
                    ctrl_tokens = None
        
        elif args.pos_mode == "multi":
            # Multi-pos structured control is complex and not needed for paper-style experiments
            safe_print("INFO: structured control in multi-pos mode is disabled by design (use pos_mode=last for paper-style).")
            ctrl_tokens = None
    
    # Log position context
    safe_print("Position context:")
    for i, dp in enumerate(dest_positions):
        ctx = get_token_context(model, dest_tokens, dp)
        sp = src_positions[i] if i < len(src_positions) else "N/A"
        safe_print(f"  [{i}] dest_pos={dp} ('{ctx['target_token']}') <-> src_pos={sp}")
    safe_print("")
    
    # Determine layers
    if args.layer_sweep:
        layers = [int(x.strip()) for x in args.layer_sweep.split(",")]
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = list(range(18, 26))  # Default sweep range
    
    # Auto-pick best layer if requested
    auto_pick_results = None
    if args.auto_pick_best_layer:
        best_layer, auto_pick_results = auto_pick_best_layer(
            model, args.prompt, args.source_prompt,
            layers, dest_positions, src_positions,
            target_id, alt_id,
            args.k_per_pos, args.alpha, args.selection_mode,
            prepend_bos
        )
        layers = [best_layer]
        safe_print("")
    
    # Determine K and alpha values
    k_values = [int(x.strip()) for x in args.sweep_k.split(",")] if args.sweep_k else [args.k_per_pos]
    alpha_values = [float(x.strip()) for x in args.sweep_alpha.split(",")] if args.sweep_alpha else [args.alpha]
    
    is_sweep = len(k_values) > 1 or len(alpha_values) > 1 or len(layers) > 1
    
    # Prepare output
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"poc3_3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Run experiments
    for layer in tqdm(layers, desc="Layers", disable=len(layers) == 1, ascii=True):
        # Get caches for this layer
        _, src_cache, _ = forward_with_cache(model, args.source_prompt, layer, prepend_bos)
        # Only compute ctrl_cache if ctrl_tokens is still valid (not disabled by position resolution)
        ctrl_cache = None
        if args.control_source_prompt and ctrl_tokens is not None:
            _, ctrl_cache, _ = forward_with_cache(model, args.control_source_prompt, layer, prepend_bos)
        
        for k in k_values:
            for alpha in alpha_values:
                result = run_experiment(
                    model, dest_tokens, src_cache, ctrl_cache,
                    layer, dest_positions, src_positions, ctrl_positions,
                    target_id, alt_id, k, alpha, args.selection_mode,
                    args.enable_permute_delta_control, args.permute_trials, args.scramble_trials,
                    args.enable_shuffle_neuron_control, args.shuffle_neuron_trials,
                    args.topk_tokens
                )
                all_results.append(result)
                
                if not result["success"]:
                    safe_print(f"L{layer} K={k} a={alpha}: FAILED - {result.get('error')}")
                    continue
                
                if is_sweep:
                    ctrl_info = ""
                    if "control_structured" in result:
                        ctrl_info += f" struct={result['control_structured']['specificity_ratio']:.1f}x"
                    if "control_permuted" in result:
                        ctrl_info += f" perm={result['control_permuted']['specificity_ratio']:.1f}x"
                    if "control_scrambled_pos" in result:
                        ctrl_info += f" scram={result['control_scrambled_pos']['specificity_ratio']:.1f}x"
                    if "control_shuffled_neurons" in result:
                        ctrl_info += f" shuf={result['control_shuffled_neurons']['specificity_ratio']:.1f}x"
                    safe_print(f"L{layer:2d} K={k:3d} a={alpha:.1f}: d_diff={result['d_logit_diff']:+.4f} p/a={result['pred_vs_actual']:.2f}{ctrl_info}")
                else:
                    # Detailed single-run output
                    safe_print(f"\n--- Layer {layer}, K={k}, alpha={alpha} ---")
                    safe_print(f"Positions: {result['n_positions']}/{result['requested_positions']}, Total neurons: {result['total_neurons']}")
                    safe_print(f"Predicted effect: {result['pred_effect']:+.6f}")
                    safe_print(f"\nBaseline:")
                    b = result["baseline"]
                    safe_print(f"  logit_diff={b['logit_diff']:+.4f} rank={b['rank_target']} margin={b['margin_to_top1']:.4f} prob={b['target_prob']:.6f}")
                    safe_print(f"  top1: {b['top1_token']!r} (logit={b['top1_logit']:.4f})")
                    safe_print(f"\nSteered:")
                    s = result["steered"]
                    safe_print(f"  logit_diff={s['logit_diff']:+.4f} rank={s['rank_target']} margin={s['margin_to_top1']:.4f} prob={s['target_prob']:.6f}")
                    safe_print(f"  top1: {s['top1_token']!r} (logit={s['top1_logit']:.4f})")
                    safe_print(f"\nDeltas:")
                    safe_print(f"  d_logit_diff = {result['d_logit_diff']:+.6f}")
                    safe_print(f"  d_margin     = {result['d_margin_to_top1']:+.6f}")
                    safe_print(f"  d_rank       = {result['d_rank_target']:+d}")
                    safe_print(f"  pred/actual  = {result['pred_vs_actual']:.3f}")
                    safe_print(f"  abs_error    = {result['abs_error']:+.6f}")
                    
                    # Per-position effects
                    if "per_position_effects" in result:
                        safe_print(f"\nPer-position effects:")
                        for pe in result["per_position_effects"]:
                            safe_print(f"  pos={pe['dest_pos']}: d_diff={pe['d_logit_diff']:+.4f} (pred={pe['pred_sum']:+.4f}, k={pe['n_neurons']})")
                    
                    # Controls
                    if "control_structured" in result:
                        c = result["control_structured"]
                        safe_print(f"\nControl (structured):")
                        safe_print(f"  d_diff={c['d_logit_diff']:+.4f} pred={c['pred_effect']:+.4f} p/a={c['pred_vs_actual']:.2f}")
                        safe_print(f"  SPECIFICITY: {c['specificity_ratio']:.1f}x")
                    if "control_permuted" in result:
                        c = result["control_permuted"]
                        safe_print(f"\nControl (permuted, n={c['n_trials']}):")
                        safe_print(f"  d_diff_mean={c['d_logit_diff_mean']:+.4f} +/- {c['d_logit_diff_std']:.4f}")
                        safe_print(f"  SPECIFICITY: {c['specificity_ratio']:.1f}x")
                    if "control_scrambled_pos" in result:
                        c = result["control_scrambled_pos"]
                        safe_print(f"\nControl (pos-scrambled, n={c['n_trials']}):")
                        safe_print(f"  d_diff_mean={c['d_logit_diff_mean']:+.4f} +/- {c['d_logit_diff_std']:.4f}")
                        safe_print(f"  SPECIFICITY: {c['specificity_ratio']:.1f}x")
                    if "control_shuffled_neurons" in result:
                        c = result["control_shuffled_neurons"]
                        safe_print(f"\nControl (shuffled-neurons, n={c['n_trials']}):")
                        safe_print(f"  d_diff_mean={c['d_logit_diff_mean']:+.4f} +/- {c['d_logit_diff_std']:.4f}")
                        safe_print(f"  SPECIFICITY: {c['specificity_ratio']:.1f}x")
                    
                    safe_print(f"\nTop-k (steered):")
                    for t in s["topk"]:
                        marker = ">>>" if t["id"] == target_id else "   "
                        safe_print(f"  {marker} {t['token']!r:>12s} logit={t['logit']:+.4f} p={t['prob']:.6f}")
    
    # Summary
    safe_print("\n" + "=" * 60)
    safe_print("SUMMARY")
    safe_print("=" * 60)
    
    successful = [r for r in all_results if r.get("success")]
    if successful:
        best = max(successful, key=lambda x: x["d_logit_diff"])
        safe_print(f"\nBest: L{best['layer']} K={best['k_per_pos']} a={best['alpha']}")
        safe_print(f"  d_logit_diff = {best['d_logit_diff']:+.6f}")
        safe_print(f"  pred/actual  = {best['pred_vs_actual']:.3f}")
        safe_print(f"  d_rank       = {best['d_rank_target']:+d}")
        safe_print(f"  d_margin     = {best['d_margin_to_top1']:+.6f}")
        
        if "control_structured" in best:
            safe_print(f"  struct_spec  = {best['control_structured']['specificity_ratio']:.1f}x")
        if "control_permuted" in best:
            safe_print(f"  perm_spec    = {best['control_permuted']['specificity_ratio']:.1f}x")
        if "control_scrambled_pos" in best:
            safe_print(f"  scram_spec   = {best['control_scrambled_pos']['specificity_ratio']:.1f}x")
        if "control_shuffled_neurons" in best:
            safe_print(f"  shuf_spec    = {best['control_shuffled_neurons']['specificity_ratio']:.1f}x")
    
    # Save report
    payload = {
        "created_at_utc": _now_utc(),
        "model_name": args.model_name,
        "dtype": args.dtype,
        "seed": args.seed,
        "prompt": args.prompt,
        "source_prompt": args.source_prompt,
        "control_source_prompt": args.control_source_prompt,
        "target_token": args.target_token,
        "target_id": target_id,
        "alt_token": args.alt_token,
        "alt_id": alt_id,
        "pos_mode": args.pos_mode,
        "dest_positions": dest_positions,
        "src_positions": src_positions,
        "dest_position_context": [get_token_context(model, dest_tokens, p) for p in dest_positions],
        "layers": layers,
        "k_values": k_values,
        "alpha_values": alpha_values,
        "selection_mode": args.selection_mode,
        "enable_permute_control": args.enable_permute_delta_control,
        "permute_trials": args.permute_trials,
        "scramble_trials": args.scramble_trials,
    }
    
    if auto_pick_results:
        payload["auto_pick_layer"] = {
            "sweep_results": auto_pick_results,
            "best_layer": layers[0],
        }
    
    payload["results"] = all_results
    
    if successful:
        best = max(successful, key=lambda x: x["d_logit_diff"])
        payload["best_result"] = {
            "layer": best["layer"],
            "k_per_pos": best["k_per_pos"],
            "alpha": best["alpha"],
            "d_logit_diff": best["d_logit_diff"],
            "pred_vs_actual": best["pred_vs_actual"],
            "abs_error": best["abs_error"],
            "d_rank_target": best["d_rank_target"],
            "d_margin_to_top1": best["d_margin_to_top1"],
        }
        if "control_structured" in best:
            payload["best_result"]["struct_specificity"] = best["control_structured"]["specificity_ratio"]
        if "control_permuted" in best:
            payload["best_result"]["perm_specificity"] = best["control_permuted"]["specificity_ratio"]
        if "control_scrambled_pos" in best:
            payload["best_result"]["scram_specificity"] = best["control_scrambled_pos"]["specificity_ratio"]
        if "control_shuffled_neurons" in best:
            payload["best_result"]["shuf_specificity"] = best["control_shuffled_neurons"]["specificity_ratio"]
    
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    
    safe_print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
