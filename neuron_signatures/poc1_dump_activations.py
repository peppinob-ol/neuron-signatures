"""
POC 1: Dump MLP neuron activations from Gemma 2 2B using TransformerLens.

Usage:
    python -m neuron_signatures.poc1_dump_activations --prompts_json docs/probe_prompts_list.json
    python -m neuron_signatures.poc1_dump_activations --single_prompt "The capital of Texas is"
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from neuron_signatures.token_sanitize import safe_print, sanitize_tokens


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

GEMMA2_N_LAYERS = 26
GEMMA2_D_MLP = 9216

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


# --------------------------------------------------------------------------- #
# Prompt loading
# --------------------------------------------------------------------------- #

def load_prompts_from_json(path: str) -> List[Dict[str, str]]:
    """
    Load prompts from a JSON file.
    
    Expects format: [{"id": "...", "text": "..."}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prompts = []
    for item in data:
        prompt_id = item.get("id", f"prompt_{len(prompts)}")
        text = item.get("text", "")
        prompts.append({"id": prompt_id, "text": text})
    
    return prompts


def load_single_prompt(text: str) -> List[Dict[str, str]]:
    """Create a single-prompt list for smoke testing."""
    return [{"id": "single_prompt", "text": text}]


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_name: str, device: str, dtype: torch.dtype):
    """
    Load model via TransformerLens HookedTransformer.
    
    Returns:
        model: HookedTransformer instance
    """
    # Import here to avoid slow import at module load
    from transformer_lens import HookedTransformer
    
    safe_print(f"Loading model: {model_name}")
    safe_print(f"  device: {device}, dtype: {dtype}")
    
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=True,
    )
    
    # Log model config
    n_layers = model.cfg.n_layers
    d_mlp = model.cfg.d_mlp
    safe_print(f"  n_layers: {n_layers}, d_mlp: {d_mlp}")
    
    return model


# --------------------------------------------------------------------------- #
# Activation extraction
# --------------------------------------------------------------------------- #

def extract_activations(
    model,
    prompt_text: str,
    prepend_bos: bool,
    save_dtype: torch.dtype,
    check_finite: bool,
) -> Tuple[torch.Tensor, List[int], List[str]]:
    """
    Run a single prompt through the model and extract MLP hook_post activations.
    
    Args:
        model: HookedTransformer instance
        prompt_text: The prompt string
        prepend_bos: Whether to prepend BOS token
        save_dtype: Dtype for the output tensor
        check_finite: Whether to check for NaN/Inf
        
    Returns:
        acts: Tensor of shape [n_layers, seq_len, d_mlp]
        token_ids: List of token IDs
        tokens_raw: List of raw token strings (for sanitization later)
    """
    n_layers = model.cfg.n_layers
    
    # Tokenize
    tokens_tensor = model.to_tokens(prompt_text, prepend_bos=prepend_bos)
    token_ids = tokens_tensor[0].tolist()
    seq_len = len(token_ids)
    
    # Decode individual tokens for display
    tokens_raw = [model.tokenizer.decode([tid]) for tid in token_ids]
    
    # Build names_filter to capture only mlp.hook_post
    def names_filter(name: str) -> bool:
        return bool(re.match(r"blocks\.\d+\.mlp\.hook_post", name))
    
    # Run forward with cache
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens_tensor,
            names_filter=names_filter,
        )
    
    # Stack activations from all layers
    layer_acts = []
    for L in range(n_layers):
        key = f"blocks.{L}.mlp.hook_post"
        # Shape: [batch=1, seq_len, d_mlp] -> [seq_len, d_mlp]
        act = cache[key][0]
        layer_acts.append(act)
    
    # Stack to [n_layers, seq_len, d_mlp]
    acts = torch.stack(layer_acts, dim=0)
    
    # Move to CPU and cast to save_dtype
    acts = acts.to(device="cpu", dtype=save_dtype)
    
    # Optional finite check
    if check_finite:
        if not torch.isfinite(acts).all():
            raise ValueError("Non-finite values detected in activations")
    
    return acts, token_ids, tokens_raw


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def run_extraction(
    prompts: List[Dict[str, str]],
    model_name: str,
    device: str,
    model_dtype: torch.dtype,
    save_dtype: torch.dtype,
    prepend_bos: bool,
    check_finite: bool,
    out_dir: Path,
) -> None:
    """
    Run activation extraction on all prompts and save results.
    """
    # Load model once
    model = load_model(model_name, device, model_dtype)
    
    n_layers = model.cfg.n_layers
    d_mlp = model.cfg.d_mlp
    
    # Validate expected architecture
    if n_layers != GEMMA2_N_LAYERS:
        safe_print(f"WARNING: n_layers={n_layers}, expected {GEMMA2_N_LAYERS}")
    if d_mlp != GEMMA2_D_MLP:
        safe_print(f"WARNING: d_mlp={d_mlp}, expected {GEMMA2_D_MLP}")
    
    # Storage
    activations_dict: Dict[str, torch.Tensor] = {}
    manifest_prompts: List[Dict[str, Any]] = []
    
    total_bytes = 0
    
    safe_print(f"\nProcessing {len(prompts)} prompt(s)...")
    
    for prompt_info in tqdm(prompts, desc="Extracting", ascii=True):
        prompt_id = prompt_info["id"]
        prompt_text = prompt_info["text"]
        
        acts, token_ids, tokens_raw = extract_activations(
            model=model,
            prompt_text=prompt_text,
            prepend_bos=prepend_bos,
            save_dtype=save_dtype,
            check_finite=check_finite,
        )
        
        # Sanitize tokens for ASCII-safe storage
        tokens_ascii = sanitize_tokens(tokens_raw)
        
        # Store activation tensor
        activations_dict[prompt_id] = acts
        
        # Compute tensor size
        tensor_bytes = acts.numel() * acts.element_size()
        total_bytes += tensor_bytes
        
        # Record manifest entry
        manifest_prompts.append({
            "probe_id": prompt_id,
            "text": prompt_text,
            "token_ids": token_ids,
            "tokens_ascii": tokens_ascii,
            "seq_len": len(token_ids),
            "activation_tensor_key": prompt_id,
            "activation_tensor_shape": list(acts.shape),
            "activation_tensor_dtype": str(save_dtype).replace("torch.", ""),
        })
    
    # Build manifest
    dtype_str = str(save_dtype).replace("torch.", "")
    model_dtype_str = str(model_dtype).replace("torch.", "")
    
    manifest = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": model_name,
        "hook_name": "blocks.*.mlp.hook_post",
        "n_layers": n_layers,
        "d_mlp": d_mlp,
        "run_device": device,
        "model_dtype": model_dtype_str,
        "save_dtype": dtype_str,
        "prepend_bos": prepend_bos,
        "prompts": manifest_prompts,
        "activations_file": "activations.pt",
    }
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)
    safe_print(f"\nSaved manifest: {manifest_path}")
    
    # Save activations
    acts_path = out_dir / "activations.pt"
    torch.save(activations_dict, acts_path)
    safe_print(f"Saved activations: {acts_path}")
    
    # Summary
    avg_seq_len = sum(p["seq_len"] for p in manifest_prompts) / len(manifest_prompts)
    safe_print(f"\n--- Summary ---")
    safe_print(f"Prompts processed: {len(prompts)}")
    safe_print(f"Average seq_len: {avg_seq_len:.1f}")
    safe_print(f"Total tensor size: {total_bytes / (1024*1024):.2f} MiB")
    safe_print(f"Output directory: {out_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="POC 1: Dump MLP neuron activations from Gemma 2 2B"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompts_json",
        type=str,
        help="Path to JSON file with [{id, text}, ...]",
    )
    input_group.add_argument(
        "--single_prompt",
        type=str,
        help="Single prompt string for quick smoke test",
    )
    
    # Output options
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: runs/poc1_YYYYMMDD_HHMMSS)",
    )
    
    # Runtime options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b",
        help="HuggingFace model name (default: google/gemma-2-2b)",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model load/runtime dtype (default: bf16)",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Activation save dtype (default: bf16)",
    )
    parser.add_argument(
        "--no_prepend_bos",
        action="store_true",
        help="Disable BOS token prepending",
    )
    parser.add_argument(
        "--check_finite",
        action="store_true",
        help="Check for NaN/Inf in activations",
    )
    
    args = parser.parse_args()
    
    # Load prompts
    if args.prompts_json:
        prompts = load_prompts_from_json(args.prompts_json)
    else:
        prompts = load_single_prompt(args.single_prompt)
    
    safe_print(f"Loaded {len(prompts)} prompt(s)")
    
    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"poc1_{timestamp}"
    
    # Convert dtype strings
    model_dtype = DTYPE_MAP[args.model_dtype]
    save_dtype = DTYPE_MAP[args.save_dtype]
    
    # Run extraction
    run_extraction(
        prompts=prompts,
        model_name=args.model_name,
        device=args.device,
        model_dtype=model_dtype,
        save_dtype=save_dtype,
        prepend_bos=not args.no_prepend_bos,
        check_finite=args.check_finite,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()

