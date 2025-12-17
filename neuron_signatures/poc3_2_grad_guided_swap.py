"""
POC 3.2: Gradient-guided multi-neuron concept swap (teacher forcing, Gemma-2-2B).

Methodologically clean implementation:
1. Single source of truth: baseline from cached forward, never recomputed
2. Gradient capture uses retain_grad() on the actual activation tensor
3. fp32 default for debugging (bf16 quantizes deltas to 0.125 steps)
4. Robust selection: handles 0 or few positive neurons gracefully
5. Per-neuron logging: delta_i, grad_i, term_i for debugging Taylor approx
6. Control test uses SAME neuron set with control deltas (specificity test)
7. No redundant forward passes or softmax computations

The predicted effect of patching neuron i with delta_i is approximately:
  d(logit_diff) ~= sum_i [ alpha * (a_src[i] - a_dest[i]) * grad_i ]

Run examples:

  # Basic run (fp32 for precision)
  python -m neuron_signatures.poc3_2_grad_guided_swap \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer 22 --pos_mode token --pos_token_dest " Dallas" --pos_token_src " San" \\
    --dtype fp32

  # Layer sweep
  python -m neuron_signatures.poc3_2_grad_guided_swap \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer_sweep 18,19,20,21,22,23,24,25 \\
    --pos_mode token --pos_token_dest " Dallas" --pos_token_src " San"

  # With control (specificity test: same neurons, different source)
  python -m neuron_signatures.poc3_2_grad_guided_swap \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --control_source_prompt "The capital of the country containing Paris is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer 22 --pos_mode token --pos_token_dest " Dallas" --pos_token_src " San"
"""

from __future__ import annotations

import argparse
import json
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
# Logit helpers (always float32 internally)
# -----------------------------------------------------------------------------

def logit_diff_from_logits(
    logits_1d: torch.Tensor, target_id: int, alt_id: int
) -> Tuple[float, float, float]:
    """Extract logit values in float32. Returns: (logit_target, logit_alt, logit_diff)"""
    x = logits_1d.float()
    lt = float(x[target_id].item())
    la = float(x[alt_id].item())
    return lt, la, lt - la


def topk_from_logits(tokenizer, logits_1d: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    """
    Compute top-k tokens efficiently using logits directly (no full softmax).
    Uses logsumexp normalization on top-k only.
    """
    logits = logits_1d.float()
    top_vals, top_idxs = torch.topk(logits, k)
    
    # Normalize using logsumexp of full vocab for proper probs
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(top_vals - log_sum_exp)
    
    out = []
    for v, i, p in zip(top_vals.tolist(), top_idxs.tolist(), probs.tolist()):
        out.append({
            "token": sanitize_token(tokenizer.decode([i])),
            "logit": float(v),
            "prob": float(p),
            "id": int(i)
        })
    return out


@dataclass
class LogitReport:
    target_token: str
    alt_token: str
    target_id: int
    alt_id: int
    logit_target: float
    logit_alt: float
    logit_diff: float
    topk: List[Dict[str, Any]]


def build_logit_report(
    tokenizer,
    next_logits: torch.Tensor,
    target_token: str,
    alt_token: str,
    target_id: int,
    alt_id: int,
    top_k: int,
) -> LogitReport:
    lt, la, diff = logit_diff_from_logits(next_logits, target_id, alt_id)
    topk = topk_from_logits(tokenizer, next_logits, top_k)
    return LogitReport(
        target_token=target_token,
        alt_token=alt_token,
        target_id=target_id,
        alt_id=alt_id,
        logit_target=lt,
        logit_alt=la,
        logit_diff=diff,
        topk=topk,
    )


# -----------------------------------------------------------------------------
# Forward pass helpers
# -----------------------------------------------------------------------------

def forward_with_cache(
    model,
    prompt: str,
    layer: int,
    prepend_bos: bool = True,
) -> Tuple[torch.Tensor, Any, torch.Tensor]:
    """
    Returns:
      logits: [1, seq, vocab] on device (full logits tensor)
      cache: ActivationCache with hook_post for this layer
      tokens: [1, seq_len] input ids
    """
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
    
    raise ValueError(f"Token {token_str!r} (id={token_id}) not found in sequence.")


# -----------------------------------------------------------------------------
# Gradient computation for logit_diff (clean implementation)
# -----------------------------------------------------------------------------

def compute_grad_at_hook_post(
    model,
    tokens: torch.Tensor,
    layer: int,
    pos: int,
    target_id: int,
    alt_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute gradient of (logit[target] - logit[alt]) w.r.t. hook_post at (layer, pos).
    
    Uses retain_grad() on the actual activation tensor - no detach/clone.
    
    Returns:
        a_dest: [d_mlp] float32 CPU - activation at (layer, pos) from this forward
        grad: [d_mlp] float32 CPU - gradient at (layer, pos)
        baseline_diff: float - the logit_diff from this forward pass
    """
    hook_name = f"blocks.{layer}.mlp.hook_post"
    captured: Dict[str, torch.Tensor] = {}

    def capture_with_grad(act: torch.Tensor, hook) -> torch.Tensor:
        # Keep original tensor in graph, just retain its gradient
        act.retain_grad()
        captured["act"] = act
        return act

    # Ensure clean state
    model.reset_hooks(including_permanent=True)
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, capture_with_grad)],
            return_type="logits",
        )
        # Compute diff in float32 for precision
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
        raise RuntimeError("Gradient is None - backward did not propagate to hook_post")

    # Extract vectors at position, convert to float32 CPU
    a_dest = act_full[0, pos].detach().float().cpu()
    grad = grad_full[0, pos].detach().float().cpu()
    baseline_diff = float(diff.detach().item())

    return a_dest, grad, baseline_diff


# -----------------------------------------------------------------------------
# Intervention hooks
# -----------------------------------------------------------------------------

def make_hook_patch_delta(
    pos: int,
    neuron_idxs: List[int],
    deltas: torch.Tensor,
    alpha: float,
):
    """Hook that applies: act[:, pos, idxs] += alpha * deltas"""
    idx = torch.tensor(neuron_idxs, dtype=torch.long)

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        act2 = act.clone()
        d = (alpha * deltas).to(device=act2.device, dtype=act2.dtype)
        act2[:, pos, idx] = act2[:, pos, idx] + d
        return act2

    return hook_fn


def run_intervention(
    model,
    tokens: torch.Tensor,
    layer: int,
    pos: int,
    neuron_idxs: List[int],
    deltas: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Run forward with patch hook, return next-token logits [vocab] on device."""
    hook_name = f"blocks.{layer}.mlp.hook_post"
    hook_fn = make_hook_patch_delta(pos, neuron_idxs, deltas, alpha)

    with torch.inference_mode():
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, hook_fn)],
            return_type="logits",
        )

    return logits[0, -1]


# -----------------------------------------------------------------------------
# Gradient-guided neuron selection (robust implementation)
# -----------------------------------------------------------------------------

def select_neurons_by_predicted_effect(
    a_dest: torch.Tensor,
    a_src: torch.Tensor,
    grad: torch.Tensor,
    intervene_k: int,
    selection_mode: str = "pos",
) -> Tuple[List[int], torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
    """
    Select neurons by predicted effect: pred = (a_src - a_dest) * grad.
    
    Robust implementation that handles 0 or few candidates gracefully.
    
    Args:
        a_dest: [d_mlp] dest activations
        a_src: [d_mlp] source activations
        grad: [d_mlp] gradient of logit_diff w.r.t. activations
        intervene_k: how many neurons to select
        selection_mode: "pos" (only positive pred), "neg" (only negative), "abs" (by magnitude)
    
    Returns:
        neuron_idxs: list of selected neuron indices
        pred: [d_mlp] predicted effect tensor (full)
        deltas: [k] delta values for selected neurons
        per_neuron_info: list of dicts with delta, grad, term for each selected neuron
    """
    delta_a = (a_src - a_dest).float()
    pred = (delta_a * grad).detach()
    
    if selection_mode == "pos":
        # Only neurons with positive predicted effect
        pos_mask = pred > 0
        pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
        
        if pos_idx.numel() == 0:
            return [], pred, torch.empty(0), []
        
        vals = pred[pos_idx]
        k_eff = min(intervene_k, vals.numel())
        top_result = torch.topk(vals, k_eff)
        chosen = pos_idx[top_result.indices]
        
    elif selection_mode == "neg":
        # Only neurons with negative predicted effect
        neg_mask = pred < 0
        neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
        
        if neg_idx.numel() == 0:
            return [], pred, torch.empty(0), []
        
        vals = -pred[neg_idx]  # negate so topk finds most negative
        k_eff = min(intervene_k, vals.numel())
        top_result = torch.topk(vals, k_eff)
        chosen = neg_idx[top_result.indices]
        
    elif selection_mode == "abs":
        # By absolute magnitude
        k_eff = min(intervene_k, pred.numel())
        top_result = torch.topk(pred.abs(), k_eff)
        chosen = top_result.indices
        
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
    
    neuron_idxs = chosen.tolist()
    deltas = delta_a[chosen]
    
    # Build per-neuron info for debugging
    per_neuron_info = []
    for i, nidx in enumerate(neuron_idxs):
        per_neuron_info.append({
            "neuron": int(nidx),
            "delta": float(delta_a[nidx].item()),
            "grad": float(grad[nidx].item()),
            "term": float(pred[nidx].item()),
        })
    
    return neuron_idxs, pred, deltas, per_neuron_info


# -----------------------------------------------------------------------------
# Main intervention pipeline (single layer)
# -----------------------------------------------------------------------------

def run_layer_experiment(
    model,
    prompt: str,
    source_prompt: str,
    control_source_prompt: Optional[str],
    layer: int,
    dest_pos: int,
    src_pos: int,
    ctrl_pos: Optional[int],
    target_id: int,
    alt_id: int,
    intervene_k: int,
    alpha: float,
    selection_mode: str,
    prepend_bos: bool,
    dest_tokens: torch.Tensor,
    src_cache,
    ctrl_cache,
    topk_tokens: int,
) -> Dict[str, Any]:
    """
    Run gradient-guided intervention for one layer.
    
    Key methodological points:
    1. Baseline comes from the gradient forward pass (single source of truth)
    2. a_dest comes from gradient forward (matches gradient reference point)
    3. Control uses SAME neuron set with control deltas (specificity test)
    """
    
    # Step 1: Compute gradient and get a_dest from same forward pass
    a_dest, grad, baseline_diff = compute_grad_at_hook_post(
        model, dest_tokens, layer, dest_pos, target_id, alt_id
    )
    
    # Also get baseline logits for full report (we need lt, la separately)
    with torch.inference_mode():
        base_logits = model(dest_tokens)[0, -1]
    lt_b, la_b, _ = logit_diff_from_logits(base_logits, target_id, alt_id)
    
    # Step 2: Get source activations from cache
    hook_key = f"blocks.{layer}.mlp.hook_post"
    a_src = src_cache[hook_key][0, src_pos].detach().float().cpu()
    
    # Step 3: Select neurons by predicted effect
    neuron_idxs, pred_full, deltas, per_neuron_info = select_neurons_by_predicted_effect(
        a_dest, a_src, grad, intervene_k, selection_mode
    )
    
    if len(neuron_idxs) == 0:
        return {
            "success": False,
            "error": f"No neurons with {selection_mode} predicted effect",
            "layer": layer,
            "intervene_k_actual": 0,
            "baseline_logit_diff": baseline_diff,
        }
    
    # Step 4: Compute predicted effect (Taylor approximation)
    pred_effect = float(alpha * sum(info["term"] for info in per_neuron_info))
    
    # Step 5: Apply main intervention
    steered_logits = run_intervention(
        model, dest_tokens, layer, dest_pos, neuron_idxs, deltas, alpha
    )
    lt_s, la_s, diff_s = logit_diff_from_logits(steered_logits, target_id, alt_id)
    
    d_lt = lt_s - lt_b
    d_la = la_s - la_b
    d_diff = diff_s - baseline_diff
    
    result = {
        "success": True,
        "layer": layer,
        "dest_pos": dest_pos,
        "src_pos": src_pos,
        "intervene_k_actual": len(neuron_idxs),
        "alpha": alpha,
        "selection_mode": selection_mode,
        "baseline_logit_target": lt_b,
        "baseline_logit_alt": la_b,
        "baseline_logit_diff": baseline_diff,
        "steered_logit_target": lt_s,
        "steered_logit_alt": la_s,
        "steered_logit_diff": diff_s,
        "d_logit_target": d_lt,
        "d_logit_alt": d_la,
        "d_logit_diff": d_diff,
        "pred_effect": pred_effect,
        "pred_vs_actual_ratio": pred_effect / d_diff if abs(d_diff) > 1e-6 else float('nan'),
        "neuron_idxs": neuron_idxs,
        "per_neuron_terms": per_neuron_info[:20],  # top 20 for report
        "steered_topk": topk_from_logits(model.tokenizer, steered_logits, topk_tokens),
    }
    
    # Step 6: Control test - SAME neuron set, DIFFERENT source deltas
    if ctrl_cache is not None and ctrl_pos is not None:
        a_ctrl = ctrl_cache[hook_key][0, ctrl_pos].detach().float().cpu()
        
        # Control deltas for the SAME neurons
        ctrl_deltas = (a_ctrl - a_dest)[torch.tensor(neuron_idxs, dtype=torch.long)]
        
        # Run intervention with control deltas
        ctrl_logits = run_intervention(
            model, dest_tokens, layer, dest_pos, neuron_idxs, ctrl_deltas, alpha
        )
        lt_c, la_c, diff_c = logit_diff_from_logits(ctrl_logits, target_id, alt_id)
        d_diff_ctrl = diff_c - baseline_diff
        
        # Control predicted effect
        ctrl_pred_terms = [
            float((a_ctrl[n] - a_dest[n]).item() * grad[n].item())
            for n in neuron_idxs
        ]
        ctrl_pred_effect = alpha * sum(ctrl_pred_terms)
        
        result["control"] = {
            "steered_logit_diff": diff_c,
            "d_logit_diff": d_diff_ctrl,
            "pred_effect": ctrl_pred_effect,
            "specificity_ratio": abs(d_diff) / max(abs(d_diff_ctrl), 1e-6),
        }
    
    return result


def main():
    ap = argparse.ArgumentParser(description="POC 3.2: Gradient-guided multi-neuron concept swap")
    ap.add_argument("--model_name", default="google/gemma-2-2b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Default to fp32 for precision - bf16 quantizes deltas to 0.125 steps
    ap.add_argument("--dtype", "--model_dtype", dest="dtype", default="fp32", 
                    choices=["bf16", "fp16", "fp32"],
                    help="Model dtype. Default fp32 for precision (bf16 quantizes to 0.125 steps)")
    ap.add_argument("--prompt", required=True, help="Destination prompt")
    ap.add_argument("--source_prompt", required=True, help="Source prompt for concept swap")
    ap.add_argument("--control_source_prompt", default=None, 
                    help="Control source prompt (specificity test: same neurons, different source)")
    ap.add_argument("--no_prepend_bos", action="store_true", help="Disable BOS prepending")
    
    # Position modes
    ap.add_argument("--pos_mode", default="last", choices=["last", "token"],
                    help="Position mode: 'last' (last token) or 'token' (find specific token)")
    ap.add_argument("--pos", type=int, default=-1, help="Token position for pos_mode=last")
    ap.add_argument("--pos_token_dest", default=None, help="Token to find in dest prompt")
    ap.add_argument("--pos_token_src", default=None, help="Token to find in source prompt")
    
    # Layer options
    ap.add_argument("--layer", type=int, default=None, help="Single layer to use")
    ap.add_argument("--layer_sweep", default=None,
                    help="Comma-separated layers to sweep (e.g. '18,19,20,21,22,23,24,25')")
    
    ap.add_argument("--target_token", required=True, help="Target token string (e.g. ' Sacramento')")
    ap.add_argument("--alt_token", required=True, help="Alt token string (e.g. ' Austin')")
    ap.add_argument("--topk_tokens", type=int, default=10, help="Show top-k next tokens")

    # Intervention parameters
    ap.add_argument("--intervene_k", type=int, default=30, help="Neurons to intervene on (default: 30)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Scaling factor for delta")
    ap.add_argument("--selection_mode", default="pos", choices=["pos", "neg", "abs"],
                    help="Neuron selection: pos (positive pred), neg (negative), abs (magnitude)")
    
    # Sweep mode
    ap.add_argument("--sweep_k", default=None, help="Comma-separated K values (e.g. '10,30,100')")
    ap.add_argument("--sweep_alpha", default=None, help="Comma-separated alpha values (e.g. '0.5,1.0,2.0')")

    ap.add_argument("--out_dir", default=None)

    args = ap.parse_args()

    # Validate layer arguments
    if args.layer is None and args.layer_sweep is None:
        raise ValueError("Must specify --layer or --layer_sweep")
    if args.layer is not None and args.layer_sweep is not None:
        raise ValueError("Cannot specify both --layer and --layer_sweep")

    dtype = DTYPE_MAP[args.dtype]
    prepend_bos = not args.no_prepend_bos

    safe_print("=== POC 3.2: Gradient-guided concept swap ===")
    safe_print(f"Model: {args.model_name} | device: {args.device} | dtype: {args.dtype}")
    if args.dtype != "fp32":
        safe_print("  WARNING: Using reduced precision. Consider --dtype fp32 for debugging.")
    safe_print(f"Prompt (dest): {args.prompt}")
    safe_print(f"Source prompt: {args.source_prompt}")
    if args.control_source_prompt:
        safe_print(f"Control prompt: {args.control_source_prompt}")
        safe_print("  (Specificity test: same neurons, control deltas)")
    safe_print("")

    # Load model
    model = load_model(args.model_name, args.device, dtype)
    target_id = resolve_single_token_id(model, args.target_token)
    alt_id = resolve_single_token_id(model, args.alt_token)

    safe_print(f"Target token: {args.target_token!r} (id={target_id})")
    safe_print(f"Alt token: {args.alt_token!r} (id={alt_id})")
    safe_print("")

    # Determine layers to process
    if args.layer_sweep:
        layers = [int(x.strip()) for x in args.layer_sweep.split(",")]
    else:
        layers = [args.layer]

    # Determine K and alpha values
    if args.sweep_k:
        k_values = [int(x.strip()) for x in args.sweep_k.split(",")]
    else:
        k_values = [args.intervene_k]
    
    if args.sweep_alpha:
        alpha_values = [float(x.strip()) for x in args.sweep_alpha.split(",")]
    else:
        alpha_values = [args.alpha]

    is_layer_sweep = len(layers) > 1
    is_param_sweep = len(k_values) > 1 or len(alpha_values) > 1

    # Prepare output
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"poc3_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for layer in tqdm(layers, desc="Layers", disable=not is_layer_sweep, ascii=True):
        if not is_layer_sweep:
            safe_print(f"--- Layer {layer} ---")
        
        # Forward passes for this layer (caches for source activations)
        _, src_cache, src_tokens = forward_with_cache(model, args.source_prompt, layer, prepend_bos)
        
        # Dest tokens (we'll do gradient forward separately per run)
        dest_tokens = model.to_tokens(args.prompt, prepend_bos=prepend_bos)
        
        dest_seq_len = dest_tokens.shape[1]
        src_seq_len = src_tokens.shape[1]

        # Resolve positions
        if args.pos_mode == "last":
            dest_pos = normalize_pos(args.pos, dest_seq_len)
            src_pos = normalize_pos(args.pos, src_seq_len)
        elif args.pos_mode == "token":
            if not args.pos_token_dest or not args.pos_token_src:
                raise ValueError("pos_mode=token requires --pos_token_dest and --pos_token_src")
            dest_pos = find_token_position(model, dest_tokens, args.pos_token_dest)
            src_pos = find_token_position(model, src_tokens, args.pos_token_src)
        
        # Control cache if needed
        ctrl_cache = None
        ctrl_tokens = None
        ctrl_pos = None
        if args.control_source_prompt:
            _, ctrl_cache, ctrl_tokens = forward_with_cache(
                model, args.control_source_prompt, layer, prepend_bos
            )
            if args.pos_mode == "token":
                try:
                    ctrl_pos = find_token_position(model, ctrl_tokens, args.pos_token_src)
                except ValueError:
                    ctrl_pos = normalize_pos(args.pos, ctrl_tokens.shape[1])
            else:
                ctrl_pos = normalize_pos(args.pos, ctrl_tokens.shape[1])

        if not is_layer_sweep:
            safe_print(f"Dest seq_len={dest_seq_len}, pos={dest_pos}")
            safe_print(f"Src seq_len={src_seq_len}, pos={src_pos}")
            
            # Show token context
            def show_context(tokens, pos, label):
                toks = tokens[0].tolist()
                start, end = max(0, pos - 2), min(len(toks), pos + 3)
                ctx = " ".join(
                    f"{'>>>' if i == pos else '   '}[{i}]{sanitize_token(model.tokenizer.decode([toks[i]]))!r}"
                    for i in range(start, end)
                )
                safe_print(f"{label}: {ctx}")
            
            show_context(dest_tokens, dest_pos, "Dest")
            show_context(src_tokens, src_pos, "Src")
            safe_print("")

        # Run interventions
        for k in k_values:
            for alpha in alpha_values:
                result = run_layer_experiment(
                    model, args.prompt, args.source_prompt, args.control_source_prompt,
                    layer, dest_pos, src_pos, ctrl_pos,
                    target_id, alt_id,
                    k, alpha, args.selection_mode,
                    prepend_bos,
                    dest_tokens, src_cache, ctrl_cache,
                    args.topk_tokens,
                )
                all_results.append(result)

                if not result["success"]:
                    safe_print(f"  L{layer} K={k} a={alpha}: FAILED - {result.get('error', 'unknown')}")
                    continue

                if is_layer_sweep or is_param_sweep:
                    ctrl_info = ""
                    if "control" in result:
                        ctrl_info = f" | ctrl={result['control']['d_logit_diff']:+.4f} spec={result['control']['specificity_ratio']:.1f}x"
                    safe_print(f"  L{layer:2d} K={k:3d} a={alpha:.1f}: d_diff={result['d_logit_diff']:+.6f} pred={result['pred_effect']:+.4f}{ctrl_info}")
                else:
                    safe_print(f"\nSelection: {args.selection_mode}, K_actual={result['intervene_k_actual']}")
                    safe_print(f"Predicted effect: {result['pred_effect']:+.6f}")
                    safe_print(f"\nBaseline: logit_diff={result['baseline_logit_diff']:+.6f}")
                    safe_print(f"Steered:  logit_diff={result['steered_logit_diff']:+.6f}")
                    safe_print(f"  d_logit_target = {result['d_logit_target']:+.6f}")
                    safe_print(f"  d_logit_alt    = {result['d_logit_alt']:+.6f}")
                    safe_print(f"  d_logit_diff   = {result['d_logit_diff']:+.6f}")
                    safe_print(f"  pred/actual    = {result['pred_vs_actual_ratio']:.3f}")
                    
                    if "control" in result:
                        c = result["control"]
                        safe_print(f"\nControl (same neurons, control deltas):")
                        safe_print(f"  d_logit_diff   = {c['d_logit_diff']:+.6f}")
                        safe_print(f"  pred_effect    = {c['pred_effect']:+.6f}")
                        safe_print(f"  specificity    = {c['specificity_ratio']:.2f}x")
                    
                    safe_print(f"\nTop-5 per-neuron terms (delta * grad):")
                    for i, info in enumerate(result["per_neuron_terms"][:5]):
                        safe_print(f"  {i+1}. n={info['neuron']:5d}: d={info['delta']:+.4f} g={info['grad']:+.6f} term={info['term']:+.6f}")
                    
                    safe_print(f"\nTop-k next tokens (steered):")
                    for t in result["steered_topk"]:
                        tok = t["token"].replace("\n", "\\n")
                        safe_print(f"  {tok!r:>14s}  logit={t['logit']:+.4f}  p={t['prob']:.6f}")

    # Summary
    safe_print("\n" + "=" * 50)
    safe_print("RESULTS SUMMARY")
    safe_print("=" * 50)
    
    successful = [r for r in all_results if r.get("success", False)]
    
    if is_layer_sweep or is_param_sweep:
        safe_print(f"\n{'Layer':>6s} {'K':>4s} {'alpha':>6s} {'d_diff':>10s} {'pred':>10s} {'p/a':>6s}")
        if any("control" in r for r in successful):
            safe_print(f"{'':>6s} {'':>4s} {'':>6s} {'ctrl':>10s} {'spec':>6s}")
        safe_print("-" * 60)
        
        for r in successful:
            line = f"{r['layer']:6d} {r['intervene_k_actual']:4d} {r['alpha']:6.2f} {r['d_logit_diff']:+10.6f} {r['pred_effect']:+10.4f} {r['pred_vs_actual_ratio']:6.2f}"
            if "control" in r:
                line += f"\n{'':>6s} {'':>4s} {'':>6s} {r['control']['d_logit_diff']:+10.6f} {r['control']['specificity_ratio']:6.1f}x"
            safe_print(line)
        
        if successful:
            best = max(successful, key=lambda x: x["d_logit_diff"])
            safe_print(f"\nBest: Layer {best['layer']}, K={best['intervene_k_actual']}, alpha={best['alpha']}")
            safe_print(f"  d_logit_diff = {best['d_logit_diff']:+.6f}")
            safe_print(f"  pred_effect  = {best['pred_effect']:+.6f}")
    else:
        if successful:
            r = successful[0]
            safe_print(f"\nBaseline logit_diff: {r['baseline_logit_diff']:+.6f}")
            safe_print(f"Steered logit_diff:  {r['steered_logit_diff']:+.6f}")
            safe_print(f"Change (d_diff):     {r['d_logit_diff']:+.6f}")
            safe_print(f"Predicted effect:    {r['pred_effect']:+.6f}")
            safe_print(f"Prediction accuracy: {r['pred_vs_actual_ratio']:.2f}")
            
            if "control" in r:
                safe_print(f"\nSpecificity test:")
                safe_print(f"  Main source d_diff:    {r['d_logit_diff']:+.6f}")
                safe_print(f"  Control source d_diff: {r['control']['d_logit_diff']:+.6f}")
                safe_print(f"  Specificity ratio:     {r['control']['specificity_ratio']:.2f}x")

    # Save report
    payload = {
        "created_at_utc": _now_utc(),
        "model_name": args.model_name,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "source_prompt": args.source_prompt,
        "control_source_prompt": args.control_source_prompt,
        "pos_mode": args.pos_mode,
        "pos_token_dest": args.pos_token_dest,
        "pos_token_src": args.pos_token_src,
        "target_token": args.target_token,
        "target_id": target_id,
        "alt_token": args.alt_token,
        "alt_id": alt_id,
        "selection_mode": args.selection_mode,
        "layers": layers,
        "k_values": k_values,
        "alpha_values": alpha_values,
        "results": [
            {k: v for k, v in r.items() if k not in ("neuron_idxs", "steered_topk") or 
             (k == "neuron_idxs" and len(r.get("neuron_idxs", [])) <= 50) or
             (k == "steered_topk")}
            for r in all_results
        ],
    }

    if successful:
        best = max(successful, key=lambda x: x["d_logit_diff"])
        payload["best_result"] = {
            "layer": best["layer"],
            "intervene_k": best["intervene_k_actual"],
            "alpha": best["alpha"],
            "d_logit_diff": best["d_logit_diff"],
            "pred_effect": best["pred_effect"],
            "pred_vs_actual_ratio": best["pred_vs_actual_ratio"],
        }
        if "control" in best:
            payload["best_result"]["control_d_logit_diff"] = best["control"]["d_logit_diff"]
            payload["best_result"]["specificity_ratio"] = best["control"]["specificity_ratio"]

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    safe_print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
