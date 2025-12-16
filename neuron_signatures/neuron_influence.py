"""
Neuron influence metrics for MLP neurons (Gemma-2 via TransformerLens).

This module implements two practical neuron scoring methods for a specific
target logit (token_id) at a specific logit position (target_pos):

1) DLA (direct, pre-ln2_post approximation):
     score ~= a(L, ctx_idx, i) * <W_out(L, i, :), W_U(:, target_token_id)>

   This is "TransformerLens-style" direct logit attribution for neurons, but it
   does NOT account for Gemma-2's post-MLP RMSNorm (ln2_post). Treat it as an
   interpretable first lens, not a faithful causal metric.

2) act_grad (activation * gradient):
     score ~= a(L, ctx_idx, i) * d logit_target / d a(L, ctx_idx, i)

   This propagates through ln2_post and downstream computation, and is often a
   better proxy for local influence than DLA.

All outputs are returned on CPU as float32 tensors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from neuron_signatures.token_sanitize import sanitize_token


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def load_model(model_name: str, device: str, dtype: torch.dtype):
    # Import lazily to avoid slow import when not needed.
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model


def _get_unembed_W_U(model) -> torch.Tensor:
    # TransformerLens typically exposes either model.W_U or model.unembed.W_U.
    if hasattr(model, "W_U"):
        return model.W_U
    if hasattr(model, "unembed") and hasattr(model.unembed, "W_U"):
        return model.unembed.W_U
    raise AttributeError("Could not find unembed weights (W_U) on model")


def _get_mlp_W_out(model, layer: int) -> torch.Tensor:
    try:
        mlp = model.blocks[layer].mlp
    except Exception as e:
        raise AttributeError(f"Could not access model.blocks[{layer}].mlp") from e

    if hasattr(mlp, "W_out"):
        return mlp.W_out
    raise AttributeError("Could not find mlp.W_out on model; TransformerLens API changed?")


def _decode_token_ascii(model, token_id: int) -> str:
    try:
        tok = model.tokenizer.decode([int(token_id)])
    except Exception:
        tok = ""
    return sanitize_token(str(tok))


def resolve_single_token_id(model, token_str: str) -> int:
    """
    Convert a user-supplied string into exactly one token id (no BOS).

    Raises if it tokenizes to multiple ids.
    """
    toks = model.to_tokens(token_str, prepend_bos=False)
    token_ids = toks[0].tolist()
    if len(token_ids) != 1:
        pieces = [sanitize_token(model.tokenizer.decode([tid])) for tid in token_ids]
        raise ValueError(
            "target_token must tokenize to exactly 1 token_id. "
            f"Got {len(token_ids)} token_ids={token_ids} pieces={pieces}"
        )
    return int(token_ids[0])


def normalize_pos(pos: int, seq_len: int) -> int:
    """Convert possibly-negative index to [0..seq_len-1]."""
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if pos < 0:
        pos = seq_len + pos
    if pos < 0 or pos >= seq_len:
        raise IndexError(f"pos out of range: pos={pos}, seq_len={seq_len}")
    return int(pos)


@dataclass(frozen=True)
class InfluenceTarget:
    target_token_id: int
    target_token_ascii: str
    target_pos: int
    ctx_idx: int


def compute_top1_token_id(
    *, model, token_ids: List[int], target_pos: int
) -> Tuple[int, str]:
    """
    Compute the argmax (top-1) token id at a given logit position for a prompt.

    This runs a forward pass on the provided prompt token_ids and returns:
      (top1_token_id, top1_token_ascii)
    """
    if not token_ids:
        raise ValueError("token_ids is empty")
    device = next(model.parameters()).device
    tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    seq_len = int(tokens_tensor.shape[1])
    tpos = normalize_pos(int(target_pos), seq_len)
    with torch.no_grad():
        logits = model(tokens_tensor)
    top1 = int(torch.argmax(logits[0, tpos], dim=-1).item())
    return top1, _decode_token_ascii(model, top1)


def build_influence_target(
    *,
    model,
    token_ids: List[int],
    tokens_ascii: List[str],
    target_mode: str,
    target_token: Optional[str],
    target_token_id: Optional[int],
    target_pos: Optional[int],
    ctx_idx: Optional[int],
) -> InfluenceTarget:
    """
    Build an InfluenceTarget for a specific prompt.

    target_mode:
      - "last_token": teacher-forced logit for the prompt's last token (pos = -2 -> token_id[-1])
      - "next_token": next-token logit after the prompt (pos defaults to -1) with user-provided target token
    """
    seq_len = len(token_ids)
    if seq_len != len(tokens_ascii):
        raise ValueError("token_ids and tokens_ascii length mismatch")

    if target_mode == "last_token":
        if seq_len < 2:
            raise ValueError("Need seq_len >= 2 for target_mode=last_token")
        tok_id = int(token_ids[-1])
        tok_ascii = str(tokens_ascii[-1])
        tpos = seq_len - 2
    elif target_mode == "next_token":
        if target_token_id is None and target_token is None:
            raise ValueError("target_mode=next_token requires --target_token or --target_token_id")
        tok_id = int(target_token_id) if target_token_id is not None else resolve_single_token_id(model, str(target_token))
        tok_ascii = _decode_token_ascii(model, tok_id)
        tpos = normalize_pos(int(target_pos) if target_pos is not None else -1, seq_len)
    elif target_mode == "top1_logit":
        # Argmax over vocab at target_pos, for this specific prompt forward pass.
        tpos = normalize_pos(int(target_pos) if target_pos is not None else -1, seq_len)
        tok_id, tok_ascii = compute_top1_token_id(model=model, token_ids=token_ids, target_pos=tpos)
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    cpos = tpos if ctx_idx is None else normalize_pos(int(ctx_idx), seq_len)
    return InfluenceTarget(
        target_token_id=tok_id,
        target_token_ascii=sanitize_token(str(tok_ascii)),
        target_pos=int(tpos),
        ctx_idx=int(cpos),
    )


def compute_dla_scores_for_prompt(
    *,
    model,
    hook_post: torch.Tensor,
    ctx_idx: int,
    target_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute DLA-like scores using hook_post and model weights (pre-ln2_post).

    Args:
        model: HookedTransformer
        hook_post: [n_layers, seq_len, d_mlp] tensor (CPU ok)
        ctx_idx: token position for neuron activations
        target_token_id: vocab id for target logit

    Returns:
        act: [n_layers, d_mlp] float32 (hook_post at ctx_idx)
        w_to_logit: [n_layers, d_mlp] float32 where w = <W_out_row, W_U_col>
        score: [n_layers, d_mlp] float32 where score = act * w_to_logit
    """
    if hook_post.ndim != 3:
        raise ValueError("hook_post must be [n_layers, seq_len, d_mlp]")

    n_layers = int(hook_post.shape[0])
    seq_len = int(hook_post.shape[1])
    ctx = normalize_pos(ctx_idx, seq_len)

    # Prepare W_U column once (keep model dtype to avoid large fp32 copies)
    W_U = _get_unembed_W_U(model)
    w_u_col = W_U[:, int(target_token_id)]

    # Align devices (weights may live on GPU)
    device = w_u_col.device

    # Activations at ctx on same device for multiplication
    act = hook_post[:, ctx, :].to(device=device, dtype=torch.float32)  # [L, N]

    w_to_logit_layers: List[torch.Tensor] = []
    for layer in range(n_layers):
        W_out = _get_mlp_W_out(model, layer)
        # Compute in model dtype (bf16/fp16) for memory efficiency, then cast small result to fp32.
        w = torch.matmul(W_out, w_u_col.to(device=W_out.device, dtype=W_out.dtype)).to(dtype=torch.float32)  # [d_mlp]
        w_to_logit_layers.append(w)

    w_to_logit = torch.stack(w_to_logit_layers, dim=0)  # [L, N]
    score = act * w_to_logit

    return act.detach().cpu(), w_to_logit.detach().cpu(), score.detach().cpu()


def compute_act_grad_scores_for_prompt(
    *,
    model,
    token_ids: List[int],
    target_pos: int,
    target_token_id: int,
    ctx_idx: int,
    name_filter: Optional[Callable[[str], bool]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Compute act*grad scores for hook_post at ctx_idx for each layer.

    Returns:
        act: [n_layers, d_mlp] float32
        grad: [n_layers, d_mlp] float32
        score: [n_layers, d_mlp] float32
        unused_layers: list of layer indices whose gradients were None (treated as zeros)
    """
    if not token_ids:
        raise ValueError("token_ids is empty")

    device = next(model.parameters()).device
    tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    seq_len = int(tokens_tensor.shape[1])
    tpos = normalize_pos(int(target_pos), seq_len)
    cpos = normalize_pos(int(ctx_idx), seq_len)

    n_layers = int(model.cfg.n_layers)
    d_mlp = int(model.cfg.d_mlp)

    if name_filter is None:
        regex = re.compile(r"blocks\.\d+\.mlp\.hook_post")

        def name_filter(name: str) -> bool:
            return bool(regex.match(name))

    # Store the actual HookPoint outputs (full [batch, pos, d_mlp] tensors).
    # IMPORTANT: we must differentiate w.r.t. these tensors, not a slice created
    # inside the hook, otherwise autograd treats the slice as "unused" (it is a
    # side-branch not used in the forward computation).
    act_full: Dict[str, torch.Tensor] = {}

    def _save_full(act: torch.Tensor, hook) -> torch.Tensor:
        # act: [batch, pos, d_mlp]
        act_full[str(hook.name)] = act
        return act

    # Ensure a clean hook state
    model.reset_hooks(including_permanent=True)
    model.zero_grad(set_to_none=True)

    model.add_hook(name_filter, _save_full, dir="fwd")
    logits = model(tokens_tensor)

    if logits.ndim != 3:
        raise ValueError(f"Unexpected logits shape: {list(logits.shape)}")

    scalar = logits[0, tpos, int(target_token_id)]

    # IMPORTANT: use autograd.grad to avoid allocating/storing gradients for all model parameters.
    keys = [f"blocks.{layer}.mlp.hook_post" for layer in range(n_layers)]
    inputs = [act_full[k] for k in keys]
    grads_tuple = torch.autograd.grad(
        scalar,
        inputs,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    # Collect in deterministic layer order
    acts: List[torch.Tensor] = []
    grads: List[torch.Tensor] = []
    unused_layers: List[int] = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.mlp.hook_post"
        if key not in act_full:
            raise KeyError(f"Missing activation for {key} (hooks did not fire?)")
        a = act_full[key].detach()[0, cpos, :].to(dtype=torch.float32)  # [d_mlp]
        g_raw = grads_tuple[layer]
        if g_raw is None:
            unused_layers.append(int(layer))
            g_raw = torch.zeros_like(act_full[key])
        g = g_raw.detach()[0, cpos, :].to(dtype=torch.float32)  # [d_mlp]
        acts.append(a)
        grads.append(g)

    model.reset_hooks(including_permanent=True)

    act_t = torch.stack(acts, dim=0)
    grad_t = torch.stack(grads, dim=0)
    score_t = act_t * grad_t

    # Sanity check shapes
    if list(act_t.shape) != [n_layers, d_mlp]:
        raise ValueError(f"Unexpected act shape: {list(act_t.shape)}")

    return act_t.cpu(), grad_t.cpu(), score_t.cpu(), unused_layers


def select_top_by_cumulative(
    abs_scores_flat: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return indices and values of the smallest prefix reaching `threshold` of total mass.

    Args:
        abs_scores_flat: 1D nonnegative tensor
        threshold: fraction in (0,1]

    Returns:
        sel_idx: 1D long indices into abs_scores_flat
        sel_vals: 1D values corresponding to sel_idx (sorted desc)
    """
    if abs_scores_flat.ndim != 1:
        raise ValueError("abs_scores_flat must be 1D")
    if threshold <= 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=abs_scores_flat.dtype)
    if threshold >= 1:
        sorted_vals, sorted_idx = torch.sort(abs_scores_flat, descending=True)
        return sorted_idx, sorted_vals

    sorted_vals, sorted_idx = torch.sort(abs_scores_flat, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    total = float(cumsum[-1].item()) if cumsum.numel() else 0.0
    if total <= 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=abs_scores_flat.dtype)

    cutoff_val = total * float(threshold)
    cutoff_idx = int(torch.searchsorted(cumsum, torch.tensor(cutoff_val, dtype=cumsum.dtype)).item())
    cutoff_idx = max(0, min(cutoff_idx, int(sorted_idx.numel()) - 1))
    k = cutoff_idx + 1
    return sorted_idx[:k], sorted_vals[:k]


