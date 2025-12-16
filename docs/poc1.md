# POC 1: Activation Dump

Minimal pipeline to extract MLP neuron activations from Gemma 2 2B using TransformerLens.

## Prerequisites

```bash
pip install -r requirements.txt
```

You also need access to the `google/gemma-2-2b` model (HuggingFace token with Gemma access).

## Usage

### Run on probe prompts JSON

```bash
python -m neuron_signatures.poc1_dump_activations --prompts_json docs/probe_prompts_list.json --out_dir runs/poc1_test
```

### Quick smoke test with a single prompt

```bash
python -m neuron_signatures.poc1_dump_activations --single_prompt "The capital of Texas is" --out_dir runs/poc1_smoke
```

### Full options

```
--prompts_json    Path to JSON file with [{"id":..., "text":...}, ...]
--single_prompt   Single prompt string (alternative to --prompts_json)
--out_dir         Output directory (default: runs/poc1_YYYYMMDD_HHMMSS)
--device          cuda or cpu (default: cuda if available)
--model_name      HuggingFace model id (default: google/gemma-2-2b)
--model_dtype     bf16, fp16, or fp32 (default: bf16)
--save_dtype      bf16, fp16, or fp32 (default: bf16)
--no_prepend_bos  Disable BOS token prepending
--check_finite    Enable NaN/Inf check on activations
```

## Output

- `manifest.json`: metadata, token_ids, tokens_ascii, tensor shapes
- `activations.pt`: dict mapping prompt_id to tensor of shape `[n_layers, seq_len, d_mlp]`

## Remote run (A40 via nodo207)

```bash
ssh nodo207
cd /path/to/neuron-signatures
python -m neuron_signatures.poc1_dump_activations --prompts_json docs/probe_prompts_list.json --out_dir runs/poc1_remote
```

Copy results back:

```bash
scp -r nodo207:/path/to/neuron-signatures/runs/poc1_remote ./runs/
```

## Memory estimates

Per prompt tensor size (bf16): `26 * seq_len * 9216 * 2` bytes = ~0.46 MiB per token.

| seq_len | Size per prompt |
|---------|-----------------|
| 128     | ~58.5 MiB       |
| 256     | ~117 MiB        |
| 512     | ~234 MiB        |
| 1024    | ~468 MiB        |

