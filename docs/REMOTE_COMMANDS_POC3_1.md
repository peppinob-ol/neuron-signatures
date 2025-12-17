# Quick Reference: POC3.1 Remote Commands

## 0) Pick GPU
```powershell
ssh nodo207 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
```
Choose a GPU with low memory/utilization (e.g., GPU 0).

---

## 1) Sync code to nodo207
```powershell
scp .\neuron_signatures\poc3_1_neuron_swap_smoketest.py nodo207:/home/giuseppe/neuron-signatures/neuron_signatures/poc3_1_neuron_swap_smoketest.py
```

Optional (only if you changed these locally):
```powershell
scp .\neuron_signatures\neuron_influence.py nodo207:/home/giuseppe/neuron-signatures/neuron_signatures/neuron_influence.py
scp .\neuron_signatures\token_sanitize.py nodo207:/home/giuseppe/neuron-signatures/neuron_signatures/token_sanitize.py
```

---

## 2) Run POC3.1 (GPU 0 example)

### Default run (10 neurons, alpha=1.0)
```powershell
ssh nodo207 "cd /home/giuseppe/neuron-signatures; export CUDA_VISIBLE_DEVICES=0; ./.venv/bin/python -m neuron_signatures.poc3_1_neuron_swap_smoketest --prompt 'The capital of the state containing Dallas is' --source_prompt 'The capital of the state containing San Francisco is' --target_token ' Sacramento' --alt_token ' Austin' --layer 22 --pos -1 --screen_topk_neurons 2000 --screen_eval_k 50 --intervene_k 10 --alpha 1.0 --intervention_mode patch_delta_from_source --out_dir runs/poc3_1_gpu0"
```

### Stronger intervention (20 neurons, alpha=2.0)
```powershell
ssh nodo207 "cd /home/giuseppe/neuron-signatures; export CUDA_VISIBLE_DEVICES=0; ./.venv/bin/python -m neuron_signatures.poc3_1_neuron_swap_smoketest --prompt 'The capital of the state containing Dallas is' --source_prompt 'The capital of the state containing San Francisco is' --target_token ' Sacramento' --alt_token ' Austin' --layer 22 --pos -1 --screen_topk_neurons 2000 --screen_eval_k 50 --intervene_k 20 --alpha 2.0 --intervention_mode patch_delta_from_source --out_dir runs/poc3_1_strong_gpu0"
```

### Ablation mode (remove Austin-supporting neurons)
```powershell
ssh nodo207 "cd /home/giuseppe/neuron-signatures; export CUDA_VISIBLE_DEVICES=0; ./.venv/bin/python -m neuron_signatures.poc3_1_neuron_swap_smoketest --prompt 'The capital of the state containing Dallas is' --source_prompt 'The capital of the state containing San Francisco is' --target_token ' Sacramento' --alt_token ' Austin' --layer 22 --pos -1 --screen_topk_neurons 2000 --screen_eval_k 50 --intervene_k 10 --intervention_mode ablate_set --out_dir runs/poc3_1_ablate_gpu0"
```

---

## 3) Retrieve results
```powershell
scp -r nodo207:/home/giuseppe/neuron-signatures/runs/poc3_1_gpu0 .\runs\
```

Or for multiple runs:
```powershell
scp -r nodo207:/home/giuseppe/neuron-signatures/runs/poc3_1_strong_gpu0 .\runs\
scp -r nodo207:/home/giuseppe/neuron-signatures/runs/poc3_1_ablate_gpu0 .\runs\
```

---

## 4) Check results locally
Open `runs/poc3_1_gpu0/report.json` and look for:
```json
{
  "steered": {
    "d_logit_diff": +1.625  // <-- KEY METRIC (positive = toward Sacramento)
  }
}
```

---

## Quick parameter reference

| Flag | Default | What it does |
|------|---------|--------------|
| `--screen_topk_neurons` | 2000 | DLA proxy candidate pool |
| `--screen_eval_k` | 50 | How many to ablation-test |
| `--intervene_k` | 10 | How many neurons to intervene on |
| `--alpha` | 1.0 | Scaling for delta interventions |
| `--intervention_mode` | `patch_delta_from_source` | `ablate_set`, `add_delta_set`, or `patch_delta_from_source` |

---

## Troubleshooting

### If you see `zsh:1: parse error near ';'`
- You forgot to replace `<GPU_ID>` with an actual number (e.g., `0`)

### If you see `>>` prompt in PowerShell
- Press **Ctrl+C** and re-paste the command (PowerShell thinks a quote is unclosed)

### If `d_logit_diff ≈ 0` (no effect)
- Try `--intervene_k 20 --alpha 2.0`
- Try different layer: `--layer 20` or `--layer 24`
- Check `ablation_screen_top10` in report: if all values are tiny (<0.1), this layer/pos may not be causal

