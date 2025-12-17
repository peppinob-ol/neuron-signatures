"""
POC 3: Neuron steering smoke test (teacher forcing) for Gemma-2-2B in TransformerLens.

What this validates:
1) You can causally intervene on a single MLP neuron activation: blocks.L.mlp.hook_post[..., i]
2) Next-token logits move in the expected direction for influential neurons.
3) Patch-from-source works as a primitive for "concept swap" in dense neuron space.

Run examples:
  python -m neuron_signatures.poc3_neuron_steering_smoketest `
    --prompt "The capital of the state containing Dallas is" `
    --target_token " Austin" `
    --alt_token " Sacramento" `
    --layer 22 --pos -1 --topk_neurons 20 --pick_rank 0

  python -m neuron_signatures.poc3_neuron_steering_smoketest `
    --prompt "The capital of the state containing Dallas is" `
    --source_prompt "The capital of the state containing San Francisco is" `
    --target_token " Austin" `
    --alt_token " Sacramento" `
    --layer 22 --pos -1 --topk_neurons 50 --pick_rank 0 --do_patch_from_source
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from neuron_signatures.neuron_influence import DTYPE_MAP, load_model, normalize_pos, resolve_single_token_id
from neuron_signatures.token_sanitize import safe_print, sanitize_token


@dataclass(frozen=True)
class LogitReport:
    target_token: str
    alt_token: str
    target_id: int
    alt_id: int
    logit_target: float
    logit_alt: float
    logit_diff: float
    topk: List[Dict[str, Any]]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_unembed_W_U(model) -> torch.Tensor:
    # Prefer the shared helper when possible, but keep this local (POC script).
    if hasattr(model, "W_U"):
        return model.W_U
    if hasattr(model, "unembed") and hasattr(model.unembed, "W_U"):
        return model.unembed.W_U
    raise AttributeError("Could not find unembed weights (W_U) on model")


def single_token_id(model, s: str) -> int:
    """
    Returns the token id for a string that must correspond to exactly one token.
    """
    return resolve_single_token_id(model, str(s))


def topk_from_logits(tokenizer, logits_1d: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    """
    Return top-k tokens by logit, along with normalized probabilities.

    Uses logsumexp to avoid materializing a full softmax vector (important for large vocabs).
    """
    if logits_1d.ndim != 1:
        raise ValueError(f"logits_1d must be 1D [vocab], got shape={list(logits_1d.shape)}")

    k_eff = int(min(int(k), int(logits_1d.numel())))
    logits_f = logits_1d.float()

    top_vals, top_idxs = torch.topk(logits_f, k_eff)
    log_z = torch.logsumexp(logits_f, dim=-1)
    top_probs = torch.exp(top_vals - log_z)

    out: List[Dict[str, Any]] = []
    for p, i in zip(top_probs.tolist(), top_idxs.tolist()):
        tok = ""
        try:
            tok = tokenizer.decode([int(i)])
        except Exception:
            tok = ""
        tok_ascii = sanitize_token(str(tok))
        out.append({"token": tok_ascii, "prob": float(p), "id": int(i)})
    return out


def forward_next_logits_with_cache(
    model,
    prompt: str,
    layer: int,
    prepend_bos: bool = True,
) -> Tuple[torch.Tensor, Any, torch.Tensor]:
    """
    Returns:
      next_logits: [vocab] logits for the next token (at the last position)
      cache: ActivationCache
      tokens: [1, seq_len] input ids
    """
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    hook_key = f"blocks.{int(layer)}.mlp.hook_post"

    def names_filter(name: str) -> bool:
        return name == hook_key

    with torch.inference_mode():
        logits, cache = model.run_with_cache(tokens, names_filter=names_filter, return_type="logits")

    next_logits = logits[0, -1]
    return next_logits, cache, tokens


def compute_neuron_dla_scores(
    model,
    cache,
    layer: int,
    pos: int,
    target_id: int,
) -> torch.Tensor:
    """
    DLA-like score per neuron at (layer,pos):
      score_i = a_i * dot(W_out[i], W_U[:, target_id])

    This ignores Gemma-2 ln2_post (RMSNorm) and mediation; it's a selection proxy for the smoke test.

    Returns:
      scores: [d_mlp] on CPU float32
    """
    hook_key = f"blocks.{int(layer)}.mlp.hook_post"
    post = cache[hook_key]  # [1, seq, d_mlp] on model device
    a = post[0, int(pos)].float()  # [d_mlp]

    W_out = model.blocks[int(layer)].mlp.W_out  # [d_mlp, d_model]
    W_U = _get_unembed_W_U(model)

    # Support either [d_model, vocab] (TL typical) or [vocab, d_model] (fallback).
    if W_U.ndim != 2:
        raise ValueError(f"Unexpected W_U ndim: {W_U.ndim}")
    if int(W_U.shape[0]) == int(model.cfg.d_model):
        w_tok = W_U[:, int(target_id)]
    elif int(W_U.shape[1]) == int(model.cfg.d_model):
        w_tok = W_U[int(target_id), :]
    else:
        raise ValueError(f"Unexpected W_U shape: {list(W_U.shape)} vs d_model={int(model.cfg.d_model)}")

    w_tok = w_tok.to(device=W_out.device, dtype=W_out.dtype)
    write_to_tok = torch.matmul(W_out, w_tok).float()  # [d_mlp]
    scores = (a.to(device=write_to_tok.device) * write_to_tok).detach().cpu()
    return scores


def make_hook_modify_single_neuron(
    layer: int,
    pos: int,
    neuron_idx: int,
    mode: str,
    *,
    add_delta: float = 0.0,
    set_value: Optional[float] = None,
    patch_value: Optional[float] = None,
):
    """
    Returns a hook function for blocks.{layer}.mlp.hook_post.

    Modes:
      - "ablate": set activation to 0
      - "flip": a := -a
      - "add": a := a + add_delta
      - "set": a := set_value (must be provided)
      - "patch": a := patch_value (must be provided)
    """

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        # act: [batch, seq, d_mlp]
        act2 = act.clone()
        p = int(pos)
        i = int(neuron_idx)

        if mode == "ablate":
            act2[:, p, i] = act2.new_tensor(0.0)
        elif mode == "flip":
            act2[:, p, i] = -act2[:, p, i]
        elif mode == "add":
            act2[:, p, i] = act2[:, p, i] + act2.new_tensor(float(add_delta))
        elif mode == "set":
            if set_value is None:
                raise ValueError("set_value is required for mode='set'")
            act2[:, p, i] = act2.new_tensor(float(set_value))
        elif mode == "patch":
            if patch_value is None:
                raise ValueError("patch_value is required for mode='patch'")
            act2[:, p, i] = act2.new_tensor(float(patch_value))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return act2

    return hook_fn


def run_teacher_forced_with_single_hook(
    model,
    prompt: str,
    layer: int,
    hook_fn,
    prepend_bos: bool = True,
) -> torch.Tensor:
    """
    Runs a forward pass with the given hook on blocks.{layer}.mlp.hook_post.
    Returns next-token logits [vocab].
    """
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    hook_name = f"blocks.{int(layer)}.mlp.hook_post"
    with torch.inference_mode():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)], return_type="logits")
    return logits[0, -1]


def build_logit_report(
    model,
    next_logits: torch.Tensor,
    target_token: str,
    alt_token: str,
    target_id: int,
    alt_id: int,
    top_k: int,
) -> LogitReport:
    logit_target = float(next_logits[int(target_id)].item())
    logit_alt = float(next_logits[int(alt_id)].item())
    topk = topk_from_logits(model.tokenizer, next_logits, int(top_k))
    return LogitReport(
        target_token=str(target_token),
        alt_token=str(alt_token),
        target_id=int(target_id),
        alt_id=int(alt_id),
        logit_target=logit_target,
        logit_alt=logit_alt,
        logit_diff=(logit_target - logit_alt),
        topk=topk,
    )


def _report_dict_with_deltas(rep: LogitReport, base: LogitReport) -> Dict[str, Any]:
    d: Dict[str, Any] = asdict(rep)
    d["delta_vs_baseline"] = {
        "logit_target": float(rep.logit_target - base.logit_target),
        "logit_alt": float(rep.logit_alt - base.logit_alt),
        "logit_diff": float(rep.logit_diff - base.logit_diff),
    }
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="POC 3: Neuron steering smoke test (teacher forcing)")

    ap.add_argument("--model_name", default="google/gemma-2-2b")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    ap.add_argument(
        "--dtype",
        "--model_dtype",
        dest="model_dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model dtype (default: bf16).",
    )
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--source_prompt", default=None, help="Optional: for patch-from-source.")
    ap.add_argument("--no_prepend_bos", action="store_true", help="Disable BOS token prepending.")

    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pos", type=int, default=-1, help="Token position (negative allowed, -1 = last token).")

    ap.add_argument(
        "--topk_neurons",
        type=int,
        default=50,
        help="Compute DLA proxy and keep top-k by |score| (default: 50).",
    )
    ap.add_argument("--pick_rank", type=int, default=0, help="Pick which neuron among top-k (0 = top-1).")

    ap.add_argument("--target_token", required=True, help="String that tokenizes to exactly one token (e.g. ' Austin').")
    ap.add_argument("--alt_token", required=True, help="String that tokenizes to exactly one token (e.g. ' Sacramento').")

    ap.add_argument("--topk_tokens", type=int, default=10, help="Show top-k next tokens (default: 10).")
    ap.add_argument("--k_std", type=float, default=3.0, help="Delta for 'add': k * std(layer,pos) (default: 3.0).")
    ap.add_argument("--do_patch_from_source", action="store_true", default=False)
    ap.add_argument("--out_dir", default=None)

    args = ap.parse_args()

    if args.model_dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {args.model_dtype}")

    model = load_model(args.model_name, device=args.device, dtype=DTYPE_MAP[args.model_dtype])

    target_id = single_token_id(model, args.target_token)
    alt_id = single_token_id(model, args.alt_token)

    prepend_bos = not bool(args.no_prepend_bos)

    # Baseline pass + cache for scoring
    base_logits, base_cache, base_tokens = forward_next_logits_with_cache(
        model=model,
        prompt=str(args.prompt),
        layer=int(args.layer),
        prepend_bos=prepend_bos,
    )

    seq_len = int(base_tokens.shape[1])
    pos = normalize_pos(int(args.pos), seq_len)

    base_report = build_logit_report(
        model=model,
        next_logits=base_logits,
        target_token=str(args.target_token),
        alt_token=str(args.alt_token),
        target_id=target_id,
        alt_id=alt_id,
        top_k=int(args.topk_tokens),
    )

    # DLA-like scoring to pick a promising neuron
    scores = compute_neuron_dla_scores(model, base_cache, int(args.layer), pos, int(target_id))  # [d_mlp] cpu
    abs_scores = scores.abs()

    k_neurons = int(min(int(args.topk_neurons), int(abs_scores.numel())))
    top_vals, top_idxs = torch.topk(abs_scores, k=k_neurons)
    ranked = list(zip(top_idxs.tolist(), top_vals.tolist(), scores[top_idxs].tolist()))

    pick_rank = int(args.pick_rank)
    if pick_rank < 0 or pick_rank >= len(ranked):
        raise ValueError(f"pick_rank={pick_rank} out of range for topk_neurons={len(ranked)}")

    neuron_idx, score_abs, score_signed = ranked[pick_rank]

    # Current activation value and std for add-delta
    hook_key = f"blocks.{int(args.layer)}.mlp.hook_post"
    a_vec = base_cache[hook_key][0, pos].detach().float().cpu()
    a0 = float(a_vec[int(neuron_idx)].item())
    std = float(a_vec.std(unbiased=False).item())
    add_delta = float(args.k_std) * std

    # Optional patch value from a source prompt
    patch_value: Optional[float] = None
    if bool(args.do_patch_from_source):
        if not args.source_prompt:
            raise ValueError("--do_patch_from_source requires --source_prompt")
        src_logits, src_cache, src_tokens = forward_next_logits_with_cache(
            model=model,
            prompt=str(args.source_prompt),
            layer=int(args.layer),
            prepend_bos=prepend_bos,
        )
        src_seq_len = int(src_tokens.shape[1])
        src_pos = normalize_pos(int(args.pos), src_seq_len)
        patch_value = float(src_cache[hook_key][0, src_pos, int(neuron_idx)].detach().float().cpu().item())

    def run_one(mode: str, **kwargs) -> LogitReport:
        hook_fn = make_hook_modify_single_neuron(int(args.layer), pos, int(neuron_idx), mode, **kwargs)
        steered_logits = run_teacher_forced_with_single_hook(
            model=model,
            prompt=str(args.prompt),
            layer=int(args.layer),
            hook_fn=hook_fn,
            prepend_bos=prepend_bos,
        )
        return build_logit_report(
            model=model,
            next_logits=steered_logits,
            target_token=str(args.target_token),
            alt_token=str(args.alt_token),
            target_id=target_id,
            alt_id=alt_id,
            top_k=int(args.topk_tokens),
        )

    # Identity sanity: add 0.0 should match baseline closely
    rep_add0 = run_one("add", add_delta=0.0)

    rep_ablate = run_one("ablate")
    rep_flip = run_one("flip")
    rep_add = run_one("add", add_delta=add_delta)
    rep_set = run_one("set", set_value=0.0)
    rep_patch = run_one("patch", patch_value=patch_value) if patch_value is not None else None

    # Print report (ASCII-only)
    safe_print("")
    safe_print("=== POC3 Neuron Steering Smoke Test ===")
    safe_print(f"model: {args.model_name} | layer={int(args.layer)} | pos={int(pos)} | neuron={int(neuron_idx)}")
    safe_print(f"DLA_proxy score signed={float(score_signed):+.6g} | abs={float(score_abs):.6g}")
    safe_print(f"activation a0={a0:+.6f} | std(layer,pos)={std:.6f} | add_delta={add_delta:+.6f}")
    if patch_value is not None:
        safe_print(f"patch_value(from source)={float(patch_value):+.6f}")

    def show(label: str, rep: LogitReport) -> None:
        safe_print("")
        safe_print(f"--- {label} ---")
        d_target = rep.logit_target - base_report.logit_target
        d_alt = rep.logit_alt - base_report.logit_alt
        d_diff = rep.logit_diff - base_report.logit_diff
        safe_print(
            f"logit({rep.target_token})={rep.logit_target:+.4f} | "
            f"logit({rep.alt_token})={rep.logit_alt:+.4f} | diff={rep.logit_diff:+.4f}"
        )
        safe_print(f"d_logit_target={d_target:+.4f} | d_logit_alt={d_alt:+.4f} | d_logit_diff={d_diff:+.4f}")
        safe_print("top-k next token probs:")
        for r in rep.topk[: int(args.topk_tokens)]:
            tok = str(r.get("token", "")).replace("\n", "\\n")
            safe_print(f"  {tok!r:>14s}  p={float(r.get('prob', 0.0)):.4f}")

    show("BASELINE", base_report)
    show("ADD(0.0) sanity", rep_add0)
    show("ABLATE", rep_ablate)
    show("FLIP", rep_flip)
    show(f"ADD(k*std), k={float(args.k_std)}", rep_add)
    show("SET(0.0)", rep_set)
    if rep_patch is not None:
        show("PATCH(from source)", rep_patch)

    # Save JSON
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"poc3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "created_at_utc": _now_utc(),
        "model_name": str(args.model_name),
        "prompt": str(args.prompt),
        "source_prompt": (None if args.source_prompt is None else str(args.source_prompt)),
        "layer": int(args.layer),
        "pos": int(pos),
        "neuron_idx": int(neuron_idx),
        "dla_proxy_score_signed": float(score_signed),
        "a0": float(a0),
        "std_layer_pos": float(std),
        "add_delta": float(add_delta),
        "patch_value": (None if patch_value is None else float(patch_value)),
        "baseline": _report_dict_with_deltas(base_report, base_report),
        "add0_sanity": {
            "activation_before": float(a0),
            "activation_after": float(a0),
            **_report_dict_with_deltas(rep_add0, base_report),
        },
        "ablate": {
            "activation_before": float(a0),
            "activation_after": 0.0,
            **_report_dict_with_deltas(rep_ablate, base_report),
        },
        "flip": {
            "activation_before": float(a0),
            "activation_after": float(-a0),
            **_report_dict_with_deltas(rep_flip, base_report),
        },
        "add": {
            "activation_before": float(a0),
            "activation_after": float(a0 + add_delta),
            **_report_dict_with_deltas(rep_add, base_report),
        },
        "set0": {
            "activation_before": float(a0),
            "activation_after": 0.0,
            **_report_dict_with_deltas(rep_set, base_report),
        },
        "patch": (
            None
            if rep_patch is None
            else {
                "activation_before": float(a0),
                "activation_after": float(patch_value) if patch_value is not None else None,
                **_report_dict_with_deltas(rep_patch, base_report),
            }
        ),
        "top_neurons_by_abs_dla_proxy": [
            {"neuron": int(i), "abs": float(a), "signed": float(s)} for (i, a, s) in ranked
        ],
    }

    out_path = out_dir / "report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    safe_print("")
    safe_print(f"[OK] Saved report: {out_path.as_posix()}")


if __name__ == "__main__":
    main()


