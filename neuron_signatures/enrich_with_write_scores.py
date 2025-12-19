"""
Enrich influence ranking data with neuron write scores.

Write scores indicate what tokens a neuron "pushes" via W_out[i] @ W_U.
This is a structural proxy (ignores LayerNorm) but useful for hypothesis generation.

Usage:
    python -m neuron_signatures.enrich_with_write_scores \
        --influence_csv runs/poc1_seedprompt_gpu6/analysis/influence_seed_prompt_act_grad_top1_logit_pos8_tok22605.csv \
        --top_n 50 --topk_write 15

Output: Creates a JSON file with write scores alongside the input CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from neuron_signatures.neuron_influence import DTYPE_MAP, load_model
from neuron_signatures.token_sanitize import safe_print, sanitize_token


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_top_neurons_from_csv(
    csv_path: Path,
    top_n: int,
) -> List[Dict[str, Any]]:
    """
    Load the top N neurons from an influence CSV file.
    
    The CSV is assumed to be sorted by abs_influence (descending).
    """
    neurons = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= top_n:
                break
            neurons.append({
                "layer": int(row["layer"]),
                "neuron": int(row["neuron"]),
                "activation": float(row["activation"]),
                "influence": float(row["influence"]),
                "abs_influence": float(row["abs_influence"]),
                "prompt_id": row.get("prompt_id", ""),
                "target_token_id": int(row.get("target_token_id", 0)),
                "target_token_ascii": row.get("target_token_ascii", ""),
                "target_pos": int(row.get("target_pos", 0)),
                "metric": row.get("metric", ""),
                "grad": float(row.get("grad", 0.0)) if row.get("grad") else None,
            })
    return neurons


def compute_write_scores_batch(
    model,
    layer_neuron_pairs: List[tuple],
    topk: int,
    explicit_token_ids: Optional[List[int]] = None,
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    Compute write scores for multiple neurons efficiently.
    
    Groups neurons by layer to minimize weight fetches.
    
    Args:
        model: HookedTransformer
        layer_neuron_pairs: List of (layer, neuron_idx) tuples
        topk: Number of top pos/neg tokens to return
        explicit_token_ids: Optional list of token IDs to score explicitly
    
    Returns:
        Nested dict: result[layer][neuron_idx] = {"top_pos": [...], "top_neg": [...], ...}
    """
    W_U = getattr(model, "W_U", None)
    if W_U is None:
        raise RuntimeError("Model has no W_U (unembed) available")
    
    WU = W_U.detach().float()  # [d_model, vocab]
    device = WU.device
    
    # Group by layer
    by_layer: Dict[int, List[int]] = {}
    for layer, neuron in layer_neuron_pairs:
        by_layer.setdefault(layer, []).append(neuron)
    
    result: Dict[int, Dict[int, Dict[str, Any]]] = {}
    
    for layer, neuron_idxs in by_layer.items():
        try:
            W_out = model.blocks[layer].mlp.W_out.detach()  # [d_mlp, d_model]
        except Exception as e:
            safe_print(f"WARNING: Cannot access W_out for layer {layer}: {e}")
            continue
        
        # Get rows for selected neurons
        idx_t = torch.tensor(neuron_idxs, dtype=torch.long, device=W_out.device)
        V = W_out[idx_t].float().to(device)  # [n, d_model]
        
        # Compute scores: V @ W_U -> [n, vocab]
        scores = V @ WU
        
        result[layer] = {}
        for j, nidx in enumerate(neuron_idxs):
            s = scores[j]  # [vocab]
            
            pos_vals, pos_ids = torch.topk(s, k=topk)
            neg_vals, neg_ids = torch.topk(-s, k=topk)
            
            def decode_ids(ids, vals, sign: str):
                items = []
                for tid, v in zip(ids.tolist(), vals.tolist()):
                    tok = sanitize_token(model.tokenizer.decode([tid]))
                    items.append({
                        "id": int(tid),
                        "token": tok,
                        "score": float(v) if sign == "+" else -float(v),
                    })
                return items
            
            neuron_entry = {
                "top_pos": decode_ids(pos_ids, pos_vals, "+"),
                "top_neg": decode_ids(neg_ids, neg_vals, "-"),
            }
            
            # Explicit tokens
            if explicit_token_ids:
                explicit = []
                for tid in explicit_token_ids:
                    tok = sanitize_token(model.tokenizer.decode([tid]))
                    explicit.append({
                        "id": int(tid),
                        "token": tok,
                        "score": float(s[tid].item()),
                    })
                neuron_entry["explicit"] = explicit
            
            result[layer][nidx] = neuron_entry
    
    return result


def main():
    ap = argparse.ArgumentParser(description="Enrich influence data with write scores")
    
    # Input
    ap.add_argument("--influence_csv", required=True, help="Path to influence CSV file")
    ap.add_argument("--top_n", type=int, default=50, help="Number of top neurons to enrich")
    
    # Write score config
    ap.add_argument("--topk_write", type=int, default=15, help="Top-k tokens for pos/neg write scores")
    ap.add_argument("--explicit_tokens", default=None,
                    help="Pipe-separated token strings to score explicitly, e.g. ' Austin| Sacramento| Texas'")
    
    # Model
    ap.add_argument("--model_name", default="google/gemma-2-2b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    
    # Output
    ap.add_argument("--out_path", default=None, help="Output JSON path (default: alongside CSV)")
    
    args = ap.parse_args()
    
    csv_path = Path(args.influence_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    safe_print(f"=== Enriching influence data with write scores ===")
    safe_print(f"Input: {csv_path}")
    safe_print(f"Top N: {args.top_n}")
    safe_print(f"Top-k write: {args.topk_write}")
    safe_print("")
    
    # Load top neurons
    neurons = load_top_neurons_from_csv(csv_path, args.top_n)
    safe_print(f"Loaded {len(neurons)} neurons from CSV")
    
    # Load model
    safe_print(f"Loading model: {args.model_name} ({args.dtype})...")
    dtype = DTYPE_MAP[args.dtype]
    model = load_model(args.model_name, args.device, dtype)
    
    # Parse explicit tokens
    explicit_token_ids = None
    if args.explicit_tokens:
        explicit_strs = [s.strip() for s in args.explicit_tokens.split("|") if s.strip()]
        explicit_token_ids = []
        for ts in explicit_strs:
            toks = model.to_tokens(ts, prepend_bos=False)
            if toks.shape[1] == 1:
                explicit_token_ids.append(int(toks[0, 0].item()))
            else:
                safe_print(f"WARNING: '{ts}' tokenizes to multiple tokens, skipping")
        safe_print(f"Explicit tokens: {explicit_token_ids}")
    
    # Build layer-neuron pairs
    layer_neuron_pairs = [(n["layer"], n["neuron"]) for n in neurons]
    
    # Compute write scores
    safe_print("Computing write scores...")
    write_scores = compute_write_scores_batch(
        model, layer_neuron_pairs, args.topk_write, explicit_token_ids
    )
    
    # Merge write scores into neurons
    for n in neurons:
        layer = n["layer"]
        neuron = n["neuron"]
        if layer in write_scores and neuron in write_scores[layer]:
            ws = write_scores[layer][neuron]
            n["write_scores"] = {
                "top_pos": ws["top_pos"],
                "top_neg": ws["top_neg"],
            }
            if "explicit" in ws:
                n["write_scores"]["explicit"] = ws["explicit"]
        else:
            n["write_scores"] = None
    
    # Build output
    out_data = {
        "created_at_utc": _now_utc(),
        "source_csv": str(csv_path.as_posix()),
        "top_n": args.top_n,
        "topk_write": args.topk_write,
        "explicit_tokens": args.explicit_tokens,
        "model_name": args.model_name,
        "neurons": neurons,
    }
    
    # Determine output path
    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = csv_path.parent / f"{csv_path.stem}_write_scores_top{args.top_n}.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=True)
    
    safe_print(f"\nSaved: {out_path}")
    
    # Print summary
    safe_print("\n=== Summary ===")
    for i, n in enumerate(neurons[:10]):
        ws = n.get("write_scores")
        if ws:
            top_pos_str = ", ".join([f"{t['token']}({t['score']:+.2f})" for t in ws["top_pos"][:5]])
            top_neg_str = ", ".join([f"{t['token']}({t['score']:+.2f})" for t in ws["top_neg"][:5]])
            safe_print(f"\n[{i+1}] L{n['layer']}:N{n['neuron']} (infl={n['influence']:+.4f})")
            safe_print(f"    +: {top_pos_str}")
            safe_print(f"    -: {top_neg_str}")


if __name__ == "__main__":
    main()

