"""
POC 3.1: Multi-neuron concept swap smoke test (teacher forcing, Gemma-2-2B).

Key improvements over POC3:
1. Always cast logits to float32 before computing diffs (fixes bf16 quantization false negatives)
2. Ablation-based screening: rank neurons by measured causal effect on logit_diff
3. Multi-neuron interventions: patch/add/ablate a set of neurons, not just one
4. Delta-based patching: a += alpha * (a_src - a_dest) instead of a = a_src
5. Position modes: intervene at last token OR at concept token (Dallas/SF)
6. Sign filtering: select only pos/neg/abs d_diff neurons
7. Sweep mode: grid search over intervene_k and alpha
8. Control prompt: compare against unrelated source prompt

Run examples:

  # Basic run (last token position)
  python -m neuron_signatures.poc3_1_neuron_swap_smoketest \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer 22 --pos_mode last

  # Concept token position (Dallas/SF)
  python -m neuron_signatures.poc3_1_neuron_swap_smoketest \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer 22 --pos_mode token --pos_token_dest " Dallas" --pos_token_src " San"

  # Sign filtering (only neurons that help Sacramento)
  python -m neuron_signatures.poc3_1_neuron_swap_smoketest \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer 22 --pos_mode token --pos_token_dest " Dallas" --pos_token_src " San" \\
    --screen_sign pos

  # Sweep mode (grid search)
  python -m neuron_signatures.poc3_1_neuron_swap_smoketest \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer 22 --pos_mode token --pos_token_dest " Dallas" --pos_token_src " San" \\
    --sweep_k 10,30,100 --sweep_alpha 0.5,1.0,2.0

  # Control experiment
  python -m neuron_signatures.poc3_1_neuron_swap_smoketest \\
    --prompt "The capital of the state containing Dallas is" \\
    --source_prompt "The capital of the state containing San Francisco is" \\
    --control_source_prompt "The capital of the country containing Paris is" \\
    --target_token " Sacramento" --alt_token " Austin" \\
    --layer 22 --pos_mode last
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


def topk_from_logits(tokenizer, logits_1d: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    """Compute top-k tokens from logits (must be float32 on CPU)."""
    probs = torch.softmax(logits_1d.float(), dim=-1)
    vals, idxs = torch.topk(probs, k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append({"token": sanitize_token(tokenizer.decode([i])), "prob": float(v), "id": int(i)})
    return out


def forward_next_logits_with_cache(
    model,
    prompt: str,
    layer: int,
    prepend_bos: bool = True,
) -> Tuple[torch.Tensor, Any, torch.Tensor]:
    """
    Returns:
      next_logits: [vocab] logits for next token (float32 CPU)
      cache: ActivationCache
      tokens: [1, seq_len] input ids
    """
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    key = f"blocks.{layer}.mlp.hook_post"

    def names_filter(name: str) -> bool:
        return name == key

    with torch.inference_mode():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter, return_type="logits")

    # CRITICAL: cast to float32 to avoid bf16 quantization masking small changes
    next_logits = logits[0, -1].float().cpu()
    return next_logits, cache, tokens


def find_token_position(model, tokens: torch.Tensor, token_str: str) -> int:
    """
    Find the position of a token string in the tokenized sequence.
    Returns the index of the LAST occurrence if multiple matches.
    """
    token_id = resolve_single_token_id(model, token_str)
    token_list = tokens[0].tolist()
    
    # Find last occurrence
    for i in range(len(token_list) - 1, -1, -1):
        if token_list[i] == token_id:
            return i
    
    # If exact match not found, try partial match (token might be split)
    # Decode each position and check for substring
    for i in range(len(token_list) - 1, -1, -1):
        decoded = model.tokenizer.decode([token_list[i]])
        if token_str.strip() in decoded or decoded in token_str:
            return i
    
    raise ValueError(f"Token {token_str!r} (id={token_id}) not found in sequence. "
                     f"Tokens: {[model.tokenizer.decode([t]) for t in token_list]}")


def compute_neuron_dla_scores(
    model,
    cache,
    layer: int,
    pos: int,
    target_id: int,
) -> torch.Tensor:
    """
    DLA-like score per neuron at (layer,pos):
      score_i = a_i * dot(W_out[i], W_U[target])
    Returns [d_mlp] float32 CPU.
    """
    hook_key = f"blocks.{layer}.mlp.hook_post"
    post = cache[hook_key]
    
    # CRITICAL: use no_grad and detach to avoid inference tensor issues
    with torch.no_grad():
        a = post[0, pos].detach().clone()
        
        W_out = model.blocks[layer].mlp.W_out.detach()
        W_U = model.W_U.detach()

        if W_U.shape[0] == model.cfg.d_model:
            w_tok = W_U[:, target_id]
        else:
            w_tok = W_U[target_id, :]

        write_to_tok = W_out @ w_tok
        scores = a * write_to_tok
    
    return scores.float().cpu()


def make_hook_modify_single_neuron(
    layer: int,
    pos: int,
    neuron_idx: int,
    mode: str,
):
    """Hook for single-neuron ablation (used in screening)."""
    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        act = act.clone()
        if mode == "ablate":
            act[:, pos, neuron_idx] = 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return act

    return hook_fn


def make_hook_modify_neuron_set(
    layer: int,
    pos: int,
    neuron_idxs: List[int],
    mode: str,
    *,
    add_deltas: Optional[torch.Tensor] = None,
    patch_values: Optional[torch.Tensor] = None,
):
    """
    Hook for blocks.{layer}.mlp.hook_post that edits multiple neurons at one position.
    """
    idx = torch.tensor(neuron_idxs, dtype=torch.long)

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        act2 = act.clone()
        p = int(pos)

        if mode == "ablate_set":
            act2[:, p, idx] = 0.0
        elif mode == "add_delta_set":
            if add_deltas is None:
                raise ValueError("add_deltas required for add_delta_set")
            d = add_deltas.to(device=act2.device, dtype=act2.dtype)
            act2[:, p, idx] = act2[:, p, idx] + d
        elif mode == "patch_set":
            if patch_values is None:
                raise ValueError("patch_values required for patch_set")
            v = patch_values.to(device=act2.device, dtype=act2.dtype)
            act2[:, p, idx] = v
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return act2

    return hook_fn


def run_with_hook(
    model,
    prompt: str,
    layer: int,
    hook_fn,
    prepend_bos: bool = True,
) -> torch.Tensor:
    """Run forward with hook, return next-token logits (float32 CPU)."""
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    hook_name = f"blocks.{layer}.mlp.hook_post"

    with torch.inference_mode():
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, hook_fn)],
            return_type="logits",
        )

    return logits[0, -1].float().cpu()


def build_logit_report(
    model,
    next_logits: torch.Tensor,
    target_token: str,
    alt_token: str,
    target_id: int,
    alt_id: int,
    top_k: int,
) -> LogitReport:
    """Build LogitReport from next-token logits (must be float32)."""
    logit_target = float(next_logits[target_id].item())
    logit_alt = float(next_logits[alt_id].item())
    topk = topk_from_logits(model.tokenizer, next_logits, top_k)
    return LogitReport(
        target_token=target_token,
        alt_token=alt_token,
        target_id=target_id,
        alt_id=alt_id,
        logit_target=logit_target,
        logit_alt=logit_alt,
        logit_diff=logit_target - logit_alt,
        topk=topk,
    )


def ablation_screen(
    model,
    prompt: str,
    layer: int,
    pos: int,
    candidate_idxs: List[int],
    target_id: int,
    alt_id: int,
    prepend_bos: bool,
) -> List[Tuple[int, float, float, float]]:
    """
    Screen neurons by ablation: measure d_logit_diff for each candidate.

    Returns:
        List of (neuron_idx, d_logit_diff, logit_target_ablated, logit_alt_ablated)
        sorted by abs(d_logit_diff) descending.
    """
    base_logits, _, _ = forward_next_logits_with_cache(model, prompt, layer, prepend_bos)
    base_diff = float(base_logits[target_id].item() - base_logits[alt_id].item())

    results = []
    safe_print(f"Ablation screening {len(candidate_idxs)} candidate neurons...")
    for i in tqdm(candidate_idxs, desc="Ablation screen", ascii=True):
        hook = make_hook_modify_single_neuron(layer, pos, i, "ablate")
        logits = run_with_hook(model, prompt, layer, hook, prepend_bos)
        lt = float(logits[target_id].item())
        la = float(logits[alt_id].item())
        diff = lt - la
        d_diff = diff - base_diff
        results.append((int(i), float(d_diff), lt, la))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results


def filter_by_sign(
    ablation_results: List[Tuple[int, float, float, float]],
    sign_mode: str,
) -> List[Tuple[int, float, float, float]]:
    """
    Filter ablation results by sign of d_logit_diff.
    
    sign_mode:
      - "abs": keep all, sorted by abs (default)
      - "pos": keep only d_diff > 0 (ablation helps target over alt)
      - "neg": keep only d_diff < 0 (ablation hurts target relative to alt)
    """
    if sign_mode == "abs":
        return ablation_results
    elif sign_mode == "pos":
        filtered = [(n, d, lt, la) for (n, d, lt, la) in ablation_results if d > 0]
        filtered.sort(key=lambda x: x[1], reverse=True)  # sort by d_diff descending
        return filtered
    elif sign_mode == "neg":
        filtered = [(n, d, lt, la) for (n, d, lt, la) in ablation_results if d < 0]
        filtered.sort(key=lambda x: x[1])  # sort by d_diff ascending (most negative first)
        return filtered
    else:
        raise ValueError(f"Unknown sign_mode: {sign_mode}")


def run_single_intervention(
    model,
    prompt: str,
    layer: int,
    dest_pos: int,
    src_pos: int,
    dest_cache,
    src_cache,
    intervene_idxs: List[int],
    intervention_mode: str,
    alpha: float,
    target_token: str,
    alt_token: str,
    target_id: int,
    alt_id: int,
    topk_tokens: int,
    prepend_bos: bool,
    baseline_report: LogitReport,
) -> Tuple[LogitReport, float, float, float, Optional[torch.Tensor]]:
    """
    Run a single intervention and return (report, d_logit_target, d_logit_alt, d_logit_diff, add_deltas).
    """
    hook_key = f"blocks.{layer}.mlp.hook_post"
    a_dest = dest_cache[hook_key][0, dest_pos].detach().float().cpu()
    a_src = src_cache[hook_key][0, src_pos].detach().float().cpu()

    idx_t = torch.tensor(intervene_idxs, dtype=torch.long)
    dest_vals = a_dest[idx_t]
    src_vals = a_src[idx_t]

    add_deltas = None
    if intervention_mode in ("patch_delta_from_source", "add_delta_set"):
        delta = src_vals - dest_vals
        add_deltas = alpha * delta
        hook_fn = make_hook_modify_neuron_set(
            layer, dest_pos, intervene_idxs, "add_delta_set", add_deltas=add_deltas
        )
    elif intervention_mode == "ablate_set":
        hook_fn = make_hook_modify_neuron_set(
            layer, dest_pos, intervene_idxs, "ablate_set"
        )
    else:
        raise ValueError(f"Unknown intervention_mode: {intervention_mode}")

    steered_logits = run_with_hook(model, prompt, layer, hook_fn, prepend_bos)
    steered_report = build_logit_report(
        model, steered_logits, target_token, alt_token, target_id, alt_id, topk_tokens
    )

    d_logit_target = steered_report.logit_target - baseline_report.logit_target
    d_logit_alt = steered_report.logit_alt - baseline_report.logit_alt
    d_logit_diff = steered_report.logit_diff - baseline_report.logit_diff

    return steered_report, d_logit_target, d_logit_alt, d_logit_diff, add_deltas


def main():
    ap = argparse.ArgumentParser(description="POC 3.1: Multi-neuron concept swap smoke test")
    ap.add_argument("--model_name", default="google/gemma-2-2b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", "--model_dtype", dest="dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--prompt", required=True, help="Destination prompt")
    ap.add_argument("--source_prompt", required=True, help="Source prompt for concept swap")
    ap.add_argument("--control_source_prompt", default=None, help="Control source prompt (unrelated)")
    ap.add_argument("--no_prepend_bos", action="store_true", help="Disable BOS prepending")
    ap.add_argument("--layer", type=int, required=True)
    
    # Position modes
    ap.add_argument("--pos_mode", default="last", choices=["last", "token"],
                    help="Position mode: 'last' (last token) or 'token' (find specific token)")
    ap.add_argument("--pos", type=int, default=-1, help="Token position for pos_mode=last (default: -1)")
    ap.add_argument("--pos_token_dest", default=None, help="Token to find in dest prompt (for pos_mode=token)")
    ap.add_argument("--pos_token_src", default=None, help="Token to find in source prompt (for pos_mode=token)")
    
    ap.add_argument("--target_token", required=True, help="Target token string (e.g. ' Sacramento')")
    ap.add_argument("--alt_token", required=True, help="Alt token string (e.g. ' Austin')")
    ap.add_argument("--topk_tokens", type=int, default=10, help="Show top-k next tokens")

    # Screening parameters
    ap.add_argument("--screen_topk_neurons", type=int, default=2000,
                    help="DLA proxy candidates (default: 2000)")
    ap.add_argument("--screen_eval_k", type=int, default=50,
                    help="How many candidates to ablation-test (default: 50)")
    ap.add_argument("--screen_sign", default="abs", choices=["abs", "pos", "neg"],
                    help="Filter neurons by d_diff sign: abs (default), pos, neg")

    # Intervention parameters
    ap.add_argument("--intervene_k", type=int, default=10,
                    help="How many neurons to intervene on (default: 10)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Scaling factor for delta-based interventions (default: 1.0)")
    ap.add_argument("--intervention_mode", default="patch_delta_from_source",
                    choices=["ablate_set", "add_delta_set", "patch_delta_from_source"],
                    help="Intervention mode (default: patch_delta_from_source)")

    # Sweep mode
    ap.add_argument("--sweep_k", default=None,
                    help="Comma-separated list of intervene_k values for sweep (e.g. '10,30,100')")
    ap.add_argument("--sweep_alpha", default=None,
                    help="Comma-separated list of alpha values for sweep (e.g. '0.5,1.0,2.0')")

    ap.add_argument("--out_dir", default=None)

    args = ap.parse_args()

    dtype = DTYPE_MAP[args.dtype]
    prepend_bos = not args.no_prepend_bos

    safe_print("=== POC 3.1: Multi-neuron concept swap ===")
    safe_print(f"Model: {args.model_name} | device: {args.device} | dtype: {args.dtype}")
    safe_print(f"Layer: {args.layer} | pos_mode: {args.pos_mode}")
    safe_print(f"Prompt (dest): {args.prompt}")
    safe_print(f"Source prompt: {args.source_prompt}")
    if args.control_source_prompt:
        safe_print(f"Control prompt: {args.control_source_prompt}")
    safe_print("")

    # Load model
    model = load_model(args.model_name, args.device, dtype)
    target_id = resolve_single_token_id(model, args.target_token)
    alt_id = resolve_single_token_id(model, args.alt_token)

    safe_print(f"Target token: {args.target_token!r} (id={target_id})")
    safe_print(f"Alt token: {args.alt_token!r} (id={alt_id})")
    safe_print("")

    # Forward passes to get caches and baseline logits
    safe_print("Running baseline forward passes...")
    dest_logits, dest_cache, dest_tokens = forward_next_logits_with_cache(
        model, args.prompt, args.layer, prepend_bos
    )
    src_logits, src_cache, src_tokens = forward_next_logits_with_cache(
        model, args.source_prompt, args.layer, prepend_bos
    )

    # Optional control source
    ctrl_cache = None
    ctrl_tokens = None
    ctrl_pos = None
    if args.control_source_prompt:
        _, ctrl_cache, ctrl_tokens = forward_next_logits_with_cache(
            model, args.control_source_prompt, args.layer, prepend_bos
        )

    dest_seq_len = dest_tokens.shape[1]
    src_seq_len = src_tokens.shape[1]

    # Resolve positions based on pos_mode
    if args.pos_mode == "last":
        dest_pos = normalize_pos(args.pos, dest_seq_len)
        src_pos = normalize_pos(args.pos, src_seq_len)
        if ctrl_tokens is not None:
            ctrl_pos = normalize_pos(args.pos, ctrl_tokens.shape[1])
        pos_info = f"last (dest={dest_pos}, src={src_pos})"
    elif args.pos_mode == "token":
        if not args.pos_token_dest or not args.pos_token_src:
            raise ValueError("pos_mode=token requires --pos_token_dest and --pos_token_src")
        dest_pos = find_token_position(model, dest_tokens, args.pos_token_dest)
        src_pos = find_token_position(model, src_tokens, args.pos_token_src)
        if ctrl_tokens is not None:
            # For control, use same token as source if possible, else last
            try:
                ctrl_pos = find_token_position(model, ctrl_tokens, args.pos_token_src)
            except ValueError:
                ctrl_pos = ctrl_tokens.shape[1] - 1
        pos_info = f"token (dest={dest_pos} [{args.pos_token_dest!r}], src={src_pos} [{args.pos_token_src!r}])"
    else:
        raise ValueError(f"Unknown pos_mode: {args.pos_mode}")

    safe_print(f"Dest seq_len={dest_seq_len}, Src seq_len={src_seq_len}")
    safe_print(f"Position mode: {pos_info}")
    
    # Show token context around the intervention position
    def show_token_context(tokens, pos, label):
        toks = tokens[0].tolist()
        start = max(0, pos - 2)
        end = min(len(toks), pos + 3)
        context = []
        for i in range(start, end):
            tok_str = sanitize_token(model.tokenizer.decode([toks[i]]))
            marker = ">>>" if i == pos else "   "
            context.append(f"{marker}[{i}]{tok_str!r}")
        safe_print(f"{label} context: {' '.join(context)}")
    
    show_token_context(dest_tokens, dest_pos, "Dest")
    show_token_context(src_tokens, src_pos, "Src")
    safe_print("")

    # Baseline report
    baseline_report = build_logit_report(
        model, dest_logits, args.target_token, args.alt_token, target_id, alt_id, args.topk_tokens
    )
    safe_print("--- BASELINE (dest prompt) ---")
    safe_print(f"logit({baseline_report.target_token})={baseline_report.logit_target:+.6f}")
    safe_print(f"logit({baseline_report.alt_token})={baseline_report.logit_alt:+.6f}")
    safe_print(f"logit_diff={baseline_report.logit_diff:+.6f}")
    safe_print("")

    # Step 1: DLA proxy to get candidate pool
    safe_print(f"Computing DLA proxy for top {args.screen_topk_neurons} candidates...")
    dla_scores = compute_neuron_dla_scores(model, dest_cache, args.layer, dest_pos, target_id)
    abs_dla = dla_scores.abs()
    topk_dla_vals, topk_dla_idxs = torch.topk(abs_dla, k=min(args.screen_topk_neurons, abs_dla.numel()))
    candidate_idxs = topk_dla_idxs.tolist()
    safe_print(f"Selected {len(candidate_idxs)} candidates by DLA proxy")
    safe_print("")

    # Step 2: Ablation screening
    screen_k = min(args.screen_eval_k, len(candidate_idxs))
    screen_candidates = candidate_idxs[:screen_k]
    ablation_results = ablation_screen(
        model, args.prompt, args.layer, dest_pos, screen_candidates, target_id, alt_id, prepend_bos
    )

    # Apply sign filter
    filtered_results = filter_by_sign(ablation_results, args.screen_sign)
    safe_print("")
    safe_print(f"Sign filter: {args.screen_sign} -> {len(filtered_results)} neurons remaining")
    safe_print(f"Top 10 neurons by ablation effect (d_logit_diff):")
    for rank, (nidx, d_diff, lt, la) in enumerate(filtered_results[:10]):
        safe_print(f"  {rank+1:2d}. neuron {nidx:5d}: d_diff={d_diff:+.6f} (lt={lt:+.6f}, la={la:+.6f})")
    safe_print("")

    # Check if we have enough neurons after filtering
    if len(filtered_results) < 1:
        safe_print("ERROR: No neurons passed the sign filter. Try --screen_sign abs")
        return

    # Determine sweep parameters
    if args.sweep_k:
        sweep_k_list = [int(x.strip()) for x in args.sweep_k.split(",")]
    else:
        sweep_k_list = [args.intervene_k]
    
    if args.sweep_alpha:
        sweep_alpha_list = [float(x.strip()) for x in args.sweep_alpha.split(",")]
    else:
        sweep_alpha_list = [args.alpha]

    is_sweep = len(sweep_k_list) > 1 or len(sweep_alpha_list) > 1

    # Prepare output
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"poc3_1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run interventions
    all_results = []
    
    for k in sweep_k_list:
        for alpha in sweep_alpha_list:
            intervene_k = min(k, len(filtered_results))
            intervene_idxs = [x[0] for x in filtered_results[:intervene_k]]
            
            if not is_sweep:
                safe_print(f"Selected top {intervene_k} neurons for intervention: {intervene_idxs[:10]}...")
                safe_print("")

            # Main intervention
            steered_report, d_lt, d_la, d_diff, add_deltas = run_single_intervention(
                model, args.prompt, args.layer, dest_pos, src_pos,
                dest_cache, src_cache, intervene_idxs,
                args.intervention_mode, alpha,
                args.target_token, args.alt_token, target_id, alt_id,
                args.topk_tokens, prepend_bos, baseline_report
            )

            result_entry = {
                "intervene_k": intervene_k,
                "alpha": alpha,
                "source": "main",
                "d_logit_target": d_lt,
                "d_logit_alt": d_la,
                "d_logit_diff": d_diff,
                "steered_report": steered_report,
            }
            all_results.append(result_entry)

            if not is_sweep:
                mode_desc = f"{args.intervention_mode} (alpha={alpha})"
                safe_print(f"Running intervention: {mode_desc} on {intervene_k} neurons...")
                safe_print("")
                safe_print(f"--- STEERED ({mode_desc}) ---")
                safe_print(f"logit({steered_report.target_token})={steered_report.logit_target:+.6f} (d={d_lt:+.6f})")
                safe_print(f"logit({steered_report.alt_token})={steered_report.logit_alt:+.6f} (d={d_la:+.6f})")
                safe_print(f"logit_diff={steered_report.logit_diff:+.6f} (d={d_diff:+.6f})")
                safe_print("")
                safe_print("Top-k next tokens (steered):")
                for r in steered_report.topk:
                    tok = r["token"].replace("\n", "\\n")
                    safe_print(f"  {tok!r:>14s}  p={r['prob']:.6f}")
                safe_print("")

            # Control intervention (if provided)
            if ctrl_cache is not None and ctrl_pos is not None:
                ctrl_report, ctrl_d_lt, ctrl_d_la, ctrl_d_diff, _ = run_single_intervention(
                    model, args.prompt, args.layer, dest_pos, ctrl_pos,
                    dest_cache, ctrl_cache, intervene_idxs,
                    args.intervention_mode, alpha,
                    args.target_token, args.alt_token, target_id, alt_id,
                    args.topk_tokens, prepend_bos, baseline_report
                )
                
                ctrl_entry = {
                    "intervene_k": intervene_k,
                    "alpha": alpha,
                    "source": "control",
                    "d_logit_target": ctrl_d_lt,
                    "d_logit_alt": ctrl_d_la,
                    "d_logit_diff": ctrl_d_diff,
                    "steered_report": ctrl_report,
                }
                all_results.append(ctrl_entry)

                if not is_sweep:
                    safe_print(f"--- CONTROL (same neurons, control source) ---")
                    safe_print(f"logit_diff={ctrl_report.logit_diff:+.6f} (d={ctrl_d_diff:+.6f})")
                    safe_print("")

    # Print sweep summary if applicable
    if is_sweep:
        safe_print("=== SWEEP RESULTS ===")
        safe_print(f"{'K':>6s} {'alpha':>6s} {'source':>8s} {'d_diff':>10s}")
        safe_print("-" * 35)
        for r in all_results:
            safe_print(f"{r['intervene_k']:6d} {r['alpha']:6.2f} {r['source']:>8s} {r['d_logit_diff']:+10.6f}")
        safe_print("")

    # Build payload
    hook_key = f"blocks.{args.layer}.mlp.hook_post"
    a_dest = dest_cache[hook_key][0, dest_pos].detach().float().cpu()
    a_src = src_cache[hook_key][0, src_pos].detach().float().cpu()

    # Use first intervention result for main payload
    main_result = [r for r in all_results if r["source"] == "main"][0]
    intervene_k = main_result["intervene_k"]
    intervene_idxs = [x[0] for x in filtered_results[:intervene_k]]
    
    idx_t = torch.tensor(intervene_idxs, dtype=torch.long)
    dest_vals = a_dest[idx_t]
    src_vals = a_src[idx_t]

    payload = {
        "created_at_utc": _now_utc(),
        "model_name": args.model_name,
        "prompt": args.prompt,
        "source_prompt": args.source_prompt,
        "control_source_prompt": args.control_source_prompt,
        "layer": args.layer,
        "pos_mode": args.pos_mode,
        "pos_token_dest": args.pos_token_dest,
        "pos_token_src": args.pos_token_src,
        "dest_pos": dest_pos,
        "src_pos": src_pos,
        "target_token": args.target_token,
        "target_id": target_id,
        "alt_token": args.alt_token,
        "alt_id": alt_id,
        "screen_topk_neurons": args.screen_topk_neurons,
        "screen_eval_k": screen_k,
        "screen_sign": args.screen_sign,
        "intervention_mode": args.intervention_mode,
        "baseline": {
            "logit_target": baseline_report.logit_target,
            "logit_alt": baseline_report.logit_alt,
            "logit_diff": baseline_report.logit_diff,
            "topk": baseline_report.topk,
        },
        "ablation_screen_top10": [
            {"neuron": int(n), "d_logit_diff": float(d), "logit_target": float(lt), "logit_alt": float(la)}
            for (n, d, lt, la) in filtered_results[:10]
        ],
    }

    if is_sweep:
        payload["sweep_results"] = [
            {
                "intervene_k": r["intervene_k"],
                "alpha": r["alpha"],
                "source": r["source"],
                "d_logit_target": r["d_logit_target"],
                "d_logit_alt": r["d_logit_alt"],
                "d_logit_diff": r["d_logit_diff"],
            }
            for r in all_results
        ]
    else:
        steered_report = main_result["steered_report"]
        payload["intervene_k"] = intervene_k
        payload["alpha"] = args.alpha
        payload["steered"] = {
            "logit_target": steered_report.logit_target,
            "logit_alt": steered_report.logit_alt,
            "logit_diff": steered_report.logit_diff,
            "topk": steered_report.topk,
            "d_logit_target": main_result["d_logit_target"],
            "d_logit_alt": main_result["d_logit_alt"],
            "d_logit_diff": main_result["d_logit_diff"],
        }
        payload["intervene_neurons"] = intervene_idxs
        payload["intervention_deltas"] = {
            "dest_values": dest_vals.tolist(),
            "src_values": src_vals.tolist(),
            "deltas": (src_vals - dest_vals).tolist(),
        }

        # Add control result if present
        ctrl_results = [r for r in all_results if r["source"] == "control"]
        if ctrl_results:
            ctrl_r = ctrl_results[0]
            payload["control"] = {
                "d_logit_target": ctrl_r["d_logit_target"],
                "d_logit_alt": ctrl_r["d_logit_alt"],
                "d_logit_diff": ctrl_r["d_logit_diff"],
            }

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    safe_print(f"Saved report: {report_path}")
    safe_print("")
    safe_print("=== Summary ===")
    safe_print(f"Baseline logit_diff: {baseline_report.logit_diff:+.6f}")
    if not is_sweep:
        safe_print(f"Steered logit_diff:  {main_result['steered_report'].logit_diff:+.6f}")
        safe_print(f"Change (d_logit_diff): {main_result['d_logit_diff']:+.6f}")
    else:
        best = max([r for r in all_results if r["source"] == "main"], key=lambda x: x["d_logit_diff"])
        safe_print(f"Best d_logit_diff: {best['d_logit_diff']:+.6f} (k={best['intervene_k']}, alpha={best['alpha']})")
    safe_print("")


if __name__ == "__main__":
    main()
