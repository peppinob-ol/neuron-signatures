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

## Browse activations (TUI)

Use the interactive browser (arrow keys + shortcuts):

```bash
python -m neuron_signatures.poc1_browse_activations --run_dir runs/poc1_test_gpu6
```

Inside the UI, press `?` for the full help and keybindings.

## POC 2: Analyze and rank neurons (cross-prompt)

This produces aggregated stats per (layer, neuron) and a top-K list:

```bash
python -m neuron_signatures.poc2_analyze_run --run_dir runs/poc1_test_gpu6
```

Outputs are written to:

- `runs/poc1_test_gpu6/analysis/neurons_aggregated.csv`
- `runs/poc1_test_gpu6/analysis/top_neurons.json`

### Optional: cumulative selection (instead of top-K)

You can select the smallest prefix of neurons that reaches a given cumulative
mass of the ranking score:

```bash
python -m neuron_signatures.poc2_analyze_run --run_dir runs/poc1_test_gpu6 --cumulative_threshold 0.8
```

In `--mode peaks` this applies to `mean_peak_abs`.

### Optional: influence mode (single-prompt target logit)

This mode ranks neurons for a specific prompt and a specific target logit,
supporting two metrics:

- `dla`: direct logit attribution (pre-ln2_post approximation)
- `act_grad`: activation * gradient (often a better local influence proxy)

Prompt-specific, auto top-1 target (no averaging across prompts):

```bash
# prompt-specific: analyzes ONLY this prompt (or the first if --prompt_id is omitted)
python -m neuron_signatures.poc2_analyze_run --mode influence --run_dir runs/poc1_test_gpu6 --prompt_id 0 --target_mode top1_logit --target_pos -1 --influence_metric act_grad --cumulative_threshold 0.8
```

`top1_logit` means: pick the argmax token at `(prompt_id, target_pos)` and compute influence w.r.t. that single logit (no averaging across prompts).

Teacher-forced last token (useful for probe prompts ending in `... is X`):

```bash
python -m neuron_signatures.poc2_analyze_run --mode influence --run_dir runs/poc1_test_gpu6 --prompt_id probe_1_Austin --influence_metric dla --target_mode last_token
```

Next-token target (e.g., for a seed prompt that ends with `is`):

```bash
python -m neuron_signatures.poc2_analyze_run --mode influence --run_dir runs/poc1_test_gpu6 --prompt_id probe_1_Austin --influence_metric act_grad --target_mode next_token --target_token \" Austin\" --target_pos -1
```

## POC 3: Single-neuron steering smoke test

This validates that you can causally intervene on a single MLP neuron and measure logit shifts.

```bash
python -m neuron_signatures.poc3_neuron_steering_smoketest \
  --prompt "The capital of the state containing Dallas is" \
  --target_token " Austin" \
  --alt_token " Sacramento" \
  --layer 22 --pos -1 --topk_neurons 20 --pick_rank 0
```

With patch-from-source (concept swap primitive):

```bash
python -m neuron_signatures.poc3_neuron_steering_smoketest \
  --prompt "The capital of the state containing Dallas is" \
  --source_prompt "The capital of the state containing San Francisco is" \
  --target_token " Sacramento" \
  --alt_token " Austin" \
  --layer 22 --pos -1 --topk_neurons 50 --pick_rank 0 --do_patch_from_source
```

## POC 3.1: Multi-neuron concept swap (dense steering)

This fixes POC3's quantization issues and implements multi-neuron interventions with ablation-based screening.

Key improvements:
- Always casts logits to float32 (fixes bf16 quantization false negatives)
- Ablation screening: ranks neurons by measured causal effect on logit_diff
- Multi-neuron interventions: patch/add/ablate a set of neurons
- Delta-based patching: `a += alpha * (a_src - a_dest)` instead of `a = a_src`

```bash
python -m neuron_signatures.poc3_1_neuron_swap_smoketest \
  --prompt "The capital of the state containing Dallas is" \
  --source_prompt "The capital of the state containing San Francisco is" \
  --target_token " Sacramento" \
  --alt_token " Austin" \
  --layer 22 --pos -1 \
  --screen_topk_neurons 2000 \
  --screen_eval_k 50 \
  --intervene_k 10 \
  --alpha 1.0 \
  --intervention_mode patch_delta_from_source
```

Parameters:
- `--screen_topk_neurons`: DLA proxy candidate pool (default: 2000)
- `--screen_eval_k`: How many to ablation-test (default: 50)
- `--intervene_k`: How many neurons to intervene on (default: 10)
- `--alpha`: Scaling for delta-based interventions (default: 1.0)
- `--intervention_mode`: `ablate_set`, `add_delta_set`, or `patch_delta_from_source`

## Streamlit EDA

Start the interactive app:

```bash
streamlit run neuron_signatures/eda_streamlit.py
```

In the sidebar, set `run_dir` to your run folder (for example `runs/poc1_test_gpu6`).
If you ran POC 2, the app will also use `analysis/top_neurons.json` for ranked selection.

The app includes a Neuronpedia-style token heatmap:
- Green background = positive activation
- Orange background = negative activation
- Opacity uses logarithmic scaling (same idea as Neuronpedia)

## Remote run (A40 via nodo207)

This section is written to be run from a Windows PowerShell terminal, launching commands on `nodo207`.

Assumptions:
- Remote repo path: `/home/giuseppe/neuron-signatures`
- The remote machine has multiple GPUs (A40s). You select one with `CUDA_VISIBLE_DEVICES`.
- `python` may not exist on the remote PATH; use the repo venv python: `./.venv/bin/python`.

### 0) Pick a GPU on nodo207

List GPUs and current load:

```powershell
ssh nodo207 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
```

Choose a `<GPU_ID>` (0..7) with low memory and low utilization (example: 6).

### 1) Sync inputs to nodo207

Copy the prompts file you want to run:

```powershell
scp .\docs\probe_prompts_list.json nodo207:/home/giuseppe/neuron-signatures/docs/probe_prompts_list.json
```

If you changed code locally (e.g. `poc2_analyze_run.py` / `neuron_influence.py` / POC3 scripts), and the nodo207 folder is not a git checkout, also sync the files:

```powershell
scp .\neuron_signatures\poc2_analyze_run.py nodo207:/home/giuseppe/neuron-signatures/neuron_signatures/poc2_analyze_run.py
scp .\neuron_signatures\neuron_influence.py nodo207:/home/giuseppe/neuron-signatures/neuron_signatures/neuron_influence.py
scp .\neuron_signatures\poc3_neuron_steering_smoketest.py nodo207:/home/giuseppe/neuron-signatures/neuron_signatures/poc3_neuron_steering_smoketest.py
scp .\neuron_signatures\poc3_1_neuron_swap_smoketest.py nodo207:/home/giuseppe/neuron-signatures/neuron_signatures/poc3_1_neuron_swap_smoketest.py
```

### 2) Run POC1 (dump activations) on nodo207

```powershell
ssh nodo207 "cd /home/giuseppe/neuron-signatures; export CUDA_VISIBLE_DEVICES=<GPU_ID>; ./.venv/bin/python -m neuron_signatures.poc1_dump_activations --prompts_json docs/probe_prompts_list.json --out_dir runs/poc1_seedprompt_gpu<GPU_ID> --device cuda --model_dtype bf16 --save_dtype bf16"
```

This produces:
- `runs/poc1_seedprompt_gpu<GPU_ID>/manifest.json`
- `runs/poc1_seedprompt_gpu<GPU_ID>/activations.pt`

Sanity check prompt count (expects `seed_prompt` to be present if included in the JSON):

```powershell
ssh nodo207 "cd /home/giuseppe/neuron-signatures; grep -c '\"probe_id\"' runs/poc1_seedprompt_gpu<GPU_ID>/manifest.json; grep '\"probe_id\"' runs/poc1_seedprompt_gpu<GPU_ID>/manifest.json"
```

### 3) Run POC2 on nodo207 (optional but recommended)

Cross-prompt peaks (writes `analysis/top_neurons.json`):

```powershell
ssh nodo207 "cd /home/giuseppe/neuron-signatures; ./.venv/bin/python -m neuron_signatures.poc2_analyze_run --run_dir runs/poc1_seedprompt_gpu<GPU_ID>"
```

Influence on a single prompt, auto top-1 target at `target_pos` (writes `analysis/influence_*.json` and `*_meta.json`):

```powershell
ssh nodo207 "cd /home/giuseppe/neuron-signatures; export CUDA_VISIBLE_DEVICES=<GPU_ID>; ./.venv/bin/python -m neuron_signatures.poc2_analyze_run --mode influence --run_dir runs/poc1_seedprompt_gpu<GPU_ID> --prompt_id seed_prompt --target_mode top1_logit --target_pos -1 --influence_metric act_grad --cumulative_threshold 0.8 --device cuda --model_dtype bf16"
```

### 4) Run POC3 or POC3.1 on nodo207 (optional: neuron steering)

**POC3 (single-neuron smoke test):**

```powershell
ssh nodo207 "cd /home/giuseppe/neuron-signatures; export CUDA_VISIBLE_DEVICES=<GPU_ID>; ./.venv/bin/python -m neuron_signatures.poc3_neuron_steering_smoketest --prompt 'The capital of the state containing Dallas is' --source_prompt 'The capital of the state containing San Francisco is' --target_token ' Sacramento' --alt_token ' Austin' --layer 22 --pos -1 --topk_neurons 50 --pick_rank 0 --do_patch_from_source --out_dir runs/poc3_gpu<GPU_ID>"
```

**POC3.1 (multi-neuron concept swap with ablation screening):**

```powershell
ssh nodo207 "cd /home/giuseppe/neuron-signatures; export CUDA_VISIBLE_DEVICES=<GPU_ID>; ./.venv/bin/python -m neuron_signatures.poc3_1_neuron_swap_smoketest --prompt 'The capital of the state containing Dallas is' --source_prompt 'The capital of the state containing San Francisco is' --target_token ' Sacramento' --alt_token ' Austin' --layer 22 --pos -1 --screen_topk_neurons 2000 --screen_eval_k 50 --intervene_k 10 --alpha 1.0 --intervention_mode patch_delta_from_source --out_dir runs/poc3_1_gpu<GPU_ID>"
```

### 5) Copy results back to Windows

IMPORTANT: copy the whole run folder (not just `analysis/`), otherwise you will miss `manifest.json` and `activations.pt`.

```powershell
scp -r nodo207:/home/giuseppe/neuron-signatures/runs/poc1_seedprompt_gpu<GPU_ID> .\runs\
```

For POC3/POC3.1 results:

```powershell
scp -r nodo207:/home/giuseppe/neuron-signatures/runs/poc3_gpu<GPU_ID> .\runs\
scp -r nodo207:/home/giuseppe/neuron-signatures/runs/poc3_1_gpu<GPU_ID> .\runs\
```

Note: the generic `scp -r nodo207:/path/to/...` example above is deprecated for nodo207; use the PowerShell commands in this section instead.

## Memory estimates

Per prompt tensor size (bf16): `26 * seq_len * 9216 * 2` bytes = ~0.46 MiB per token.

| seq_len | Size per prompt |
|---------|-----------------|
| 128     | ~58.5 MiB       |
| 256     | ~117 MiB        |
| 512     | ~234 MiB        |
| 1024    | ~468 MiB        |

