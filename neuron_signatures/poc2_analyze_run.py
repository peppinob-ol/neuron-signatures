"""
POC 2: Analyze a POC 1 run (manifest.json + activations.pt) and rank neurons.

This computes simple, scalable cross-prompt summaries per neuron (layer, idx):
  - mean/std/max of per-prompt peak(|activation|) across token positions
  - mode/uniques of peak token_id and peak ctx_idx across prompts
  - fraction of prompts where the peak token is classified as functional

Outputs:
  - neurons_aggregated.csv: one row per (layer, neuron)
  - top_neurons.json: top-K neurons by mean_peak_abs
  - analysis_meta.json: summary metadata

Additional mode:
  - influence: rank neurons for a specific target logit on a specific prompt,
    using either:
      - DLA (direct, pre-ln2_post approximation)
      - act_grad (activation * gradient)

    This supports selecting neurons by a cumulative-mass threshold (Circuit
    Tracing-style "cumulative influence") in addition to top-K.

Run:
  python -m neuron_signatures.poc2_analyze_run --run_dir runs/poc1_test_gpu6
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


DEFAULT_FUNCTIONAL_TOKENS = {
    # Articles / determiners
    "the",
    "a",
    "an",
    # Common function words
    "is",
    "are",
    "was",
    "were",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "as",
    "and",
    "or",
    "but",
    "with",
    "by",
    "from",
    "which",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "be",
    # Punctuation (handled separately too)
    ",",
    ".",
    ":",
    ";",
    "?",
    "!",
}

PUNCT_TOKENS = {",", ".", ":", ";", "?", "!", "(", ")", "[", "]", "{", "}", "\"", "'"}


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "manifest.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_activations(run_dir: Path) -> Dict[str, torch.Tensor]:
    path = run_dir / "activations.pt"
    return torch.load(path, map_location="cpu")


def _token_norm(token_ascii: str) -> str:
    return token_ascii.strip().lower()


def _is_functional_token(token_ascii: str, functional_set: set[str]) -> bool:
    t = _token_norm(token_ascii)
    if not t:
        return True
    if t in PUNCT_TOKENS:
        return True
    if t == "<bos>":
        return True
    return t in functional_set


def _build_token_id_to_ascii(prompts: List[Dict[str, Any]]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for p in prompts:
        token_ids = p.get("token_ids", [])
        tokens_ascii = p.get("tokens_ascii", [])
        for tid, tok in zip(token_ids, tokens_ascii):
            if tid not in mapping:
                mapping[int(tid)] = str(tok)
    return mapping


def _mode_and_counts(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-column mode, mode count, and number of unique values.

    Args:
        values: int tensor of shape [n_prompts, N]

    Returns:
        mode_vals: [N]
        mode_counts: [N]
        n_unique: [N]
    """
    if values.ndim != 2:
        raise ValueError("values must be [n_prompts, N]")

    n_prompts = int(values.shape[0])
    if n_prompts <= 0:
        raise ValueError("n_prompts must be > 0")

    mode_vals, _ = torch.mode(values, dim=0)
    mode_counts = (values == mode_vals.unsqueeze(0)).sum(dim=0)

    if n_prompts == 1:
        n_unique = torch.ones_like(mode_vals)
    else:
        sorted_vals, _ = torch.sort(values, dim=0)
        n_unique = (sorted_vals[1:] != sorted_vals[:-1]).sum(dim=0) + 1

    return mode_vals, mode_counts, n_unique


def analyze_run(
    run_dir: Path,
    out_dir: Path,
    top_k: int,
    functional_tokens: set[str],
    cumulative_threshold: Optional[float] = None,
) -> None:
    """
    Peak-based cross-prompt summaries (existing POC 2 behavior).
    """
    manifest = _load_manifest(run_dir)
    prompts: List[Dict[str, Any]] = manifest.get("prompts", [])
    if not prompts:
        raise ValueError("manifest.json has no prompts")

    activations = _load_activations(run_dir)

    # Quick integrity check
    for p in prompts:
        pid = p.get("probe_id")
        if pid not in activations:
            raise ValueError(f"activations.pt missing prompt_id={pid}")

    n_layers = int(manifest.get("n_layers", 0))
    d_mlp = int(manifest.get("d_mlp", 0))
    n_prompts = len(prompts)
    n_neurons_total = n_layers * d_mlp

    token_id_to_ascii = _build_token_id_to_ascii(prompts)

    # Per-prompt tensors to stack: [n_prompts, n_layers, d_mlp]
    max_abs_list: List[torch.Tensor] = []
    peak_ctx_list: List[torch.Tensor] = []
    peak_token_id_list: List[torch.Tensor] = []
    peak_is_func_list: List[torch.Tensor] = []

    for p in prompts:
        pid = str(p["probe_id"])
        token_ids: List[int] = [int(x) for x in p.get("token_ids", [])]
        tokens_ascii: List[str] = [str(x) for x in p.get("tokens_ascii", [])]
        if len(token_ids) != len(tokens_ascii):
            raise ValueError(f"token_ids and tokens_ascii length mismatch for {pid}")

        token_ids_t = torch.tensor(token_ids, dtype=torch.long)
        token_is_func = torch.tensor(
            [_is_functional_token(t, functional_tokens) for t in tokens_ascii],
            dtype=torch.bool,
        )

        x = activations[pid]  # [n_layers, seq_len, d_mlp] bf16 on CPU
        if int(x.shape[0]) != n_layers or int(x.shape[2]) != d_mlp:
            raise ValueError(f"Unexpected tensor shape for {pid}: {list(x.shape)}")

        # Convert to float32 for stable stats
        x_f = x.float()
        abs_x = x_f.abs()

        # Peak(|act|) over token positions
        max_abs, peak_ctx = torch.max(abs_x, dim=1)  # both [n_layers, d_mlp]

        # Token id / functional classification at peak ctx
        peak_token_id = token_ids_t[peak_ctx]
        peak_is_func = token_is_func[peak_ctx].to(dtype=torch.float32)

        max_abs_list.append(max_abs)
        peak_ctx_list.append(peak_ctx.to(dtype=torch.long))
        peak_token_id_list.append(peak_token_id.to(dtype=torch.long))
        peak_is_func_list.append(peak_is_func)

    max_abs_stack = torch.stack(max_abs_list, dim=0)  # [P,L,N]
    peak_ctx_stack = torch.stack(peak_ctx_list, dim=0)  # [P,L,N]
    peak_token_id_stack = torch.stack(peak_token_id_list, dim=0)  # [P,L,N]
    peak_is_func_stack = torch.stack(peak_is_func_list, dim=0)  # [P,L,N]

    # Aggregate metrics
    mean_peak_abs = max_abs_stack.mean(dim=0)
    std_peak_abs = max_abs_stack.std(dim=0, unbiased=False)
    max_peak_abs = max_abs_stack.max(dim=0).values
    min_peak_abs = max_abs_stack.min(dim=0).values
    frac_peak_functional = peak_is_func_stack.mean(dim=0)

    # Mode + uniqueness for token_id and ctx_idx
    peak_token_id_flat = peak_token_id_stack.reshape(n_prompts, n_neurons_total)
    peak_ctx_flat = peak_ctx_stack.reshape(n_prompts, n_neurons_total)

    mode_token_id, mode_token_count, n_unique_token_id = _mode_and_counts(peak_token_id_flat)
    mode_ctx, mode_ctx_count, n_unique_ctx = _mode_and_counts(peak_ctx_flat)

    # Build top list by mean_peak_abs (top-K or cumulative threshold)
    mean_flat = mean_peak_abs.reshape(n_neurons_total)
    if cumulative_threshold is not None:
        from neuron_signatures.neuron_influence import select_top_by_cumulative

        top_idx, top_vals = select_top_by_cumulative(mean_flat, float(cumulative_threshold))
        k = int(top_idx.numel())
    else:
        k = min(int(top_k), int(n_neurons_total))
        top_vals, top_idx = torch.topk(mean_flat, k=k, largest=True)

    def idx_to_layer_neuron(flat_idx: int) -> Tuple[int, int]:
        layer = flat_idx // d_mlp
        neuron = flat_idx % d_mlp
        return int(layer), int(neuron)

    top_neurons: List[Dict[str, Any]] = []
    for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
        layer, neuron = idx_to_layer_neuron(int(idx))
        tok_id = int(mode_token_id[idx].item())
        top_neurons.append(
            {
                "layer": layer,
                "neuron": neuron,
                "mean_peak_abs": float(v),
                "std_peak_abs": float(std_peak_abs[layer, neuron].item()),
                "max_peak_abs": float(max_peak_abs[layer, neuron].item()),
                "frac_peak_functional": float(frac_peak_functional[layer, neuron].item()),
                "mode_peak_token_id": tok_id,
                "mode_peak_token_ascii": token_id_to_ascii.get(tok_id, ""),
                "mode_peak_token_count": int(mode_token_count[idx].item()),
                "n_unique_peak_token_id": int(n_unique_token_id[idx].item()),
                "mode_peak_ctx_idx": int(mode_ctx[idx].item()),
                "mode_peak_ctx_count": int(mode_ctx_count[idx].item()),
                "n_unique_peak_ctx_idx": int(n_unique_ctx[idx].item()),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write aggregated CSV
    csv_path = out_dir / "neurons_aggregated.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer",
                "neuron",
                "mean_peak_abs",
                "std_peak_abs",
                "max_peak_abs",
                "min_peak_abs",
                "frac_peak_functional",
                "mode_peak_token_id",
                "mode_peak_token_ascii",
                "mode_peak_token_count",
                "n_unique_peak_token_id",
                "mode_peak_ctx_idx",
                "mode_peak_ctx_count",
                "n_unique_peak_ctx_idx",
            ]
        )

        mode_token_id_2d = mode_token_id.reshape(n_layers, d_mlp)
        mode_token_count_2d = mode_token_count.reshape(n_layers, d_mlp)
        n_unique_token_id_2d = n_unique_token_id.reshape(n_layers, d_mlp)
        mode_ctx_2d = mode_ctx.reshape(n_layers, d_mlp)
        mode_ctx_count_2d = mode_ctx_count.reshape(n_layers, d_mlp)
        n_unique_ctx_2d = n_unique_ctx.reshape(n_layers, d_mlp)

        for layer in range(n_layers):
            for neuron in range(d_mlp):
                tok_id = int(mode_token_id_2d[layer, neuron].item())
                writer.writerow(
                    [
                        layer,
                        neuron,
                        float(mean_peak_abs[layer, neuron].item()),
                        float(std_peak_abs[layer, neuron].item()),
                        float(max_peak_abs[layer, neuron].item()),
                        float(min_peak_abs[layer, neuron].item()),
                        float(frac_peak_functional[layer, neuron].item()),
                        tok_id,
                        token_id_to_ascii.get(tok_id, ""),
                        int(mode_token_count_2d[layer, neuron].item()),
                        int(n_unique_token_id_2d[layer, neuron].item()),
                        int(mode_ctx_2d[layer, neuron].item()),
                        int(mode_ctx_count_2d[layer, neuron].item()),
                        int(n_unique_ctx_2d[layer, neuron].item()),
                    ]
                )

    # Write top-K JSON
    top_path = out_dir / "top_neurons.json"
    with open(top_path, "w", encoding="utf-8") as f:
        json.dump(top_neurons, f, indent=2, ensure_ascii=True)

    # Write meta JSON
    meta = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": str(run_dir.as_posix()),
        "mode": "peaks",
        "n_prompts": n_prompts,
        "n_layers": n_layers,
        "d_mlp": d_mlp,
        "n_neurons_total": n_neurons_total,
        "top_k": k,
        "cumulative_threshold": (None if cumulative_threshold is None else float(cumulative_threshold)),
        "outputs": {
            "neurons_aggregated_csv": str(csv_path.name),
            "top_neurons_json": str(top_path.name),
        },
    }
    meta_path = out_dir / "analysis_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {top_path}")
    print(f"Wrote: {meta_path}")


def _resolve_prompt_selector(
    prompts: List[Dict[str, Any]], prompt_id: Optional[str]
) -> Tuple[Dict[str, Any], int, str]:
    """
    Resolve a prompt selector to a single prompt.

    - If prompt_id is None: selects prompts[0]
    - If prompt_id is a digit string (e.g. "0"): selects prompts[int(prompt_id)]
    - Otherwise: treats prompt_id as a probe_id string and searches for it

    Returns:
        prompt_dict, prompt_index, selector_repr
    """
    if not prompts:
        raise ValueError("No prompts in manifest")
    if prompt_id is None:
        return prompts[0], 0, "first_prompt"

    pid_str = str(prompt_id).strip()
    if pid_str.isdigit():
        idx = int(pid_str)
        if idx < 0 or idx >= len(prompts):
            raise IndexError(f"prompt_id index out of range: {idx} (n_prompts={len(prompts)})")
        return prompts[idx], idx, f"index:{idx}"

    for i, p in enumerate(prompts):
        if str(p.get("probe_id", "")) == pid_str:
            return p, i, f"probe_id:{pid_str}"
    raise KeyError(f"prompt_id not found in manifest (expected index or probe_id): {pid_str}")


def analyze_influence(
    *,
    run_dir: Path,
    out_dir: Path,
    prompt_id: Optional[str],
    influence_metric: str,
    target_mode: str,
    target_token: Optional[str],
    target_token_id: Optional[int],
    target_pos: Optional[int],
    ctx_idx: Optional[int],
    top_k: int,
    cumulative_threshold: Optional[float],
    model_name: Optional[str],
    device: str,
    model_dtype: str,
) -> None:
    """
    Influence-mode: rank neurons for one prompt's target logit.
    """
    from neuron_signatures.neuron_influence import (
        DTYPE_MAP,
        build_influence_target,
        compute_act_grad_scores_for_prompt,
        compute_dla_scores_for_prompt,
        load_model,
        select_top_by_cumulative,
    )

    manifest = _load_manifest(run_dir)
    prompts: List[Dict[str, Any]] = manifest.get("prompts", [])
    if not prompts:
        raise ValueError("manifest.json has no prompts")

    prompt, prompt_idx, prompt_selector = _resolve_prompt_selector(prompts, prompt_id)
    pid = str(prompt.get("probe_id", ""))
    if not pid:
        raise ValueError("Prompt missing probe_id")

    token_ids: List[int] = [int(x) for x in prompt.get("token_ids", [])]
    tokens_ascii: List[str] = [str(x) for x in prompt.get("tokens_ascii", [])]
    if len(token_ids) != len(tokens_ascii) or not token_ids:
        raise ValueError(f"Invalid token_ids/tokens_ascii for prompt_id={pid}")

    # Load activations (needed for DLA)
    activations = None
    if influence_metric == "dla":
        activations = _load_activations(run_dir)
        if pid not in activations:
            raise ValueError(f"activations.pt missing prompt_id={pid}")

    # Load model (weights/tokenizer needed for both metrics)
    model_id = str(model_name) if model_name else str(manifest.get("model_name", "google/gemma-2-2b"))
    if model_dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported model_dtype: {model_dtype}")
    model = load_model(model_id, device=device, dtype=DTYPE_MAP[model_dtype])

    target = build_influence_target(
        model=model,
        token_ids=token_ids,
        tokens_ascii=tokens_ascii,
        target_mode=str(target_mode),
        target_token=target_token,
        target_token_id=target_token_id,
        target_pos=target_pos,
        ctx_idx=ctx_idx,
    )

    if target_mode == "top1_logit":
        target_token_source = "auto_top1"
    elif target_mode == "last_token":
        target_token_source = "teacher_forced_last_token"
    else:
        if target_token_id is not None:
            target_token_source = "user_token_id"
        else:
            target_token_source = "user_token_str"

    # Compute scores
    if influence_metric == "dla":
        if activations is None:
            raise RuntimeError("Internal error: activations not loaded for DLA")
        hook_post = activations[pid]
        act, w_to_logit, score = compute_dla_scores_for_prompt(
            model=model,
            hook_post=hook_post,
            ctx_idx=target.ctx_idx,
            target_token_id=target.target_token_id,
        )
        grad = None
    elif influence_metric == "act_grad":
        act, grad, score, unused_layers = compute_act_grad_scores_for_prompt(
            model=model,
            token_ids=token_ids,
            target_pos=target.target_pos,
            target_token_id=target.target_token_id,
            ctx_idx=target.ctx_idx,
        )
        w_to_logit = None
    else:
        raise ValueError(f"Unsupported influence_metric: {influence_metric}")

    n_layers = int(score.shape[0])
    d_mlp = int(score.shape[1])
    n_neurons_total = n_layers * d_mlp

    abs_flat = score.abs().reshape(n_neurons_total)

    if cumulative_threshold is not None:
        sel_idx, sel_vals = select_top_by_cumulative(abs_flat, float(cumulative_threshold))
        top_idx = sel_idx
        top_vals = sel_vals
    else:
        k = min(int(top_k), int(n_neurons_total))
        top_vals, top_idx = torch.topk(abs_flat, k=k, largest=True)

    def idx_to_layer_neuron(flat_idx: int) -> Tuple[int, int]:
        layer = flat_idx // d_mlp
        neuron = flat_idx % d_mlp
        return int(layer), int(neuron)

    top_neurons: List[Dict[str, Any]] = []
    for v_abs, idx in zip(top_vals.tolist(), top_idx.tolist()):
        layer, neuron = idx_to_layer_neuron(int(idx))
        raw = float(score[layer, neuron].item())
        a_val = float(act[layer, neuron].item())
        rec: Dict[str, Any] = {
            "layer": layer,
            "neuron": neuron,
            "activation": a_val,
            "influence": raw,
            "abs_influence": float(v_abs),
            "prompt_id": pid,
            "target_token_id": int(target.target_token_id),
            "target_token_ascii": str(target.target_token_ascii),
            "target_pos": int(target.target_pos),
            "ctx_idx": int(target.ctx_idx),
            "metric": str(influence_metric),
            "target_mode": str(target_mode),
        }
        if w_to_logit is not None:
            rec["w_to_logit"] = float(w_to_logit[layer, neuron].item())
        if grad is not None:
            rec["grad"] = float(grad[layer, neuron].item())
        top_neurons.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Output file names (prompt-specific)
    safe_pid = pid.replace("/", "_").replace("\\", "_")
    base = (
        f"influence_{safe_pid}_{influence_metric}_{target_mode}"
        f"_pos{int(target.target_pos)}_tok{int(target.target_token_id)}"
    )

    json_path = out_dir / f"{base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(top_neurons, f, indent=2, ensure_ascii=True)

    csv_path = out_dir / f"{base}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "layer",
            "neuron",
            "activation",
            "influence",
            "abs_influence",
            "prompt_id",
            "target_token_id",
            "target_token_ascii",
            "target_pos",
            "ctx_idx",
            "metric",
            "target_mode",
        ]
        if influence_metric == "dla":
            header.append("w_to_logit")
        if influence_metric == "act_grad":
            header.append("grad")
        writer.writerow(header)
        for r in top_neurons:
            row = [
                r["layer"],
                r["neuron"],
                r["activation"],
                r["influence"],
                r["abs_influence"],
                r["prompt_id"],
                r["target_token_id"],
                r["target_token_ascii"],
                r["target_pos"],
                r["ctx_idx"],
                r["metric"],
                r["target_mode"],
            ]
            if influence_metric == "dla":
                row.append(r.get("w_to_logit", 0.0))
            if influence_metric == "act_grad":
                row.append(r.get("grad", 0.0))
            writer.writerow(row)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": str(run_dir.as_posix()),
        "mode": "influence",
        "scope": "single_prompt",
        "prompt_selector": str(prompt_selector),
        "prompt_index": int(prompt_idx),
        "prompt_id": pid,
        "metric": str(influence_metric),
        "target_mode": str(target_mode),
        "target_token_source": str(target_token_source),
        "target_token_id": int(target.target_token_id),
        "target_token_ascii": str(target.target_token_ascii),
        "target_pos": int(target.target_pos),
        "ctx_idx": int(target.ctx_idx),
        "n_layers": n_layers,
        "d_mlp": d_mlp,
        "n_neurons_total": n_neurons_total,
        "selected_count": int(top_idx.numel()),
        "top_k_arg": int(top_k),
        "cumulative_threshold": (None if cumulative_threshold is None else float(cumulative_threshold)),
        "model_name": str(model_id),
        "device": str(device),
        "model_dtype": str(model_dtype),
        "outputs": {
            "top_neurons_json": str(json_path.name),
            "top_neurons_csv": str(csv_path.name),
        },
    }
    if influence_metric == "act_grad":
        meta["act_grad_unused_layers"] = [int(x) for x in unused_layers]
        meta["act_grad_unused_layers_count"] = int(len(unused_layers))
    meta_path = out_dir / f"{base}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="POC 2: Analyze a POC 1 activation run")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory containing manifest.json and activations.pt",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="peaks",
        choices=["peaks", "influence"],
        help="Analysis mode: peaks (cross-prompt) or influence (single-prompt target logit).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <run_dir>/analysis)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Top-K neurons to export in JSON (default: 200)",
    )
    parser.add_argument(
        "--cumulative_threshold",
        type=float,
        default=None,
        help="Optional cumulative-mass threshold in (0,1]; selects the smallest prefix reaching this mass (applies to mean_peak_abs in peaks mode, abs_influence in influence mode).",
    )
    parser.add_argument(
        "--functional_tokens_json",
        type=str,
        default=None,
        help="Optional JSON file with a list of functional tokens to override defaults",
    )

    # Influence-mode options (ignored in peaks mode)
    parser.add_argument(
        "--prompt_id",
        type=str,
        default=None,
        help="Prompt selector for influence mode: either an integer index (e.g. 0) or a probe_id string (default: first prompt).",
    )
    parser.add_argument(
        "--influence_metric",
        type=str,
        default="dla",
        choices=["dla", "act_grad"],
        help="Influence metric: dla (pre-ln2_post direct logit attribution) or act_grad (activation*gradient).",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="last_token",
        choices=["last_token", "next_token", "top1_logit"],
        help=(
            "Target definition: "
            "last_token (teacher-forced last token), "
            "next_token (user-provided token after prompt), "
            "or top1_logit (auto argmax token at target_pos)."
        ),
    )
    parser.add_argument(
        "--target_token",
        type=str,
        default=None,
        help="Target token string (must tokenize to exactly one token id). Used for target_mode=next_token.",
    )
    parser.add_argument(
        "--target_token_id",
        type=int,
        default=None,
        help="Target token id (int). Used for target_mode=next_token.",
    )
    parser.add_argument(
        "--target_pos",
        type=int,
        default=None,
        help="Logit position for target_mode=next_token or top1_logit (default: -1, i.e. last token position of the prompt).",
    )
    parser.add_argument(
        "--ctx_idx",
        type=int,
        default=None,
        help="Neuron ctx_idx for influence scoring (default: same as target_pos).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for influence-mode model run (default: cuda if available).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HuggingFace model name for influence mode (default: manifest model_name).",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model dtype for influence mode (default: bf16).",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "analysis")

    functional_tokens = set(DEFAULT_FUNCTIONAL_TOKENS)
    if args.functional_tokens_json:
        with open(args.functional_tokens_json, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError("functional_tokens_json must contain a JSON list of strings")
        functional_tokens = set(str(x).strip().lower() for x in items)

    if args.mode == "peaks":
        analyze_run(
            run_dir=run_dir,
            out_dir=out_dir,
            top_k=args.top_k,
            functional_tokens=functional_tokens,
            cumulative_threshold=args.cumulative_threshold,
        )
    else:
        analyze_influence(
            run_dir=run_dir,
            out_dir=out_dir,
            prompt_id=args.prompt_id,
            influence_metric=args.influence_metric,
            target_mode=args.target_mode,
            target_token=args.target_token,
            target_token_id=args.target_token_id,
            target_pos=args.target_pos,
            ctx_idx=args.ctx_idx,
            top_k=args.top_k,
            cumulative_threshold=args.cumulative_threshold,
            model_name=args.model_name,
            device=args.device,
            model_dtype=args.model_dtype,
        )


if __name__ == "__main__":
    main()



