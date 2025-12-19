# POC3 and POC3.1: Neuron Steering Summary

## What was wrong with POC3 (single-neuron)?

Your POC3 run (`runs/poc3_20251217_041414/report.json`) showed:

1. **Quantization masking small changes**: Logits were likely printed in bf16, so small real changes appeared as 0.0
2. **Single-neuron interventions too weak**: In dense MLP space, one neuron rarely produces large logit shifts
3. **Absolute patching vs delta patching**: Setting `a = a_src` ignores context; `a += (a_src - a_dest)` is better

### Evidence from your POC3 report

- **Baseline**: `logit_diff = -6.125` (Austin strongly preferred over Sacramento)
- **ADD(k*std)**: `d_logit_diff = 0.0` (appeared to do nothing, likely quantization)
- **ABLATE**: `d_logit_diff = +0.25` (small but real movement toward Sacramento)
- **FLIP**: `d_logit_diff = +0.375` (larger effect, as expected: -2a change)
- **PATCH**: `d_logit_diff = +0.25` (same as ablate, absolute patching not ideal)

The neuron selected was **L22.4278** (DLA proxy score = -0.14), with `a0 = -1.945`.

## POC3.1: Multi-neuron concept swap

### Key improvements

1. **Float32 logits everywhere**: No more quantization false negatives
2. **Ablation-based screening**: Ranks neurons by measured `d_logit_diff`, not just DLA proxy
3. **Multi-neuron interventions**: Intervene on top-K neurons (default: 10)
4. **Delta-based patching**: `a += alpha * (a_src - a_dest)` captures directional change

### How it works

```
1. DLA proxy → candidate pool (e.g. top 2000 neurons)
2. Ablation screen → rank by causal effect on logit_diff (test top 50)
3. Select top K neurons (e.g. 10)
4. Compute delta = a_src - a_dest for those neurons
5. Intervene: a += alpha * delta
6. Measure logit shifts in float32
```

### Expected behavior

- **Small alpha (0.5-1.0)**: Gentle nudge, should see +0.5 to +2.0 shift in logit_diff
- **Large alpha (2.0-5.0)**: Stronger push, may saturate or trigger RMSNorm nonlinearities
- **Ablate set**: Removes "Austin-supporting" neurons, should reduce Austin preference
- **Patch delta**: Adds "Sacramento-supporting" pattern from SF prompt

## Remote commands (PowerShell → nodo207)

### 1) Sync code
```powershell
scp .\neuron_signatures\poc3_1_neuron_swap_smoketest.py nodo207:/home/giuseppe/neuron-signatures/neuron_signatures/poc3_1_neuron_swap_smoketest.py
```

### 2) Run POC3.1 (example: GPU 0)
```powershell
ssh nodo207 "cd /home/giuseppe/neuron-signatures; export CUDA_VISIBLE_DEVICES=0; ./.venv/bin/python -m neuron_signatures.poc3_1_neuron_swap_smoketest --prompt 'The capital of the state containing Dallas is' --source_prompt 'The capital of the state containing San Francisco is' --target_token ' Sacramento' --alt_token ' Austin' --layer 22 --pos -1 --screen_topk_neurons 2000 --screen_eval_k 50 --intervene_k 10 --alpha 1.0 --intervention_mode patch_delta_from_source --out_dir runs/poc3_1_gpu0"
```

### 3) Retrieve results
```powershell
scp -r nodo207:/home/giuseppe/neuron-signatures/runs/poc3_1_gpu0 .\runs\
```

## Interpreting results

### What to look for in `report.json`

```json
{
  "baseline": {
    "logit_diff": -6.125  // Austin strongly preferred
  },
  "steered": {
    "logit_diff": -4.5,   // Moved toward Sacramento
    "d_logit_diff": +1.625  // THIS IS THE KEY METRIC
  },
  "ablation_screen_top10": [
    {"neuron": 4278, "d_logit_diff": +0.25},  // Ablating this helps Sacramento
    ...
  ]
}
```

### Success criteria

- **`d_logit_diff > +0.5`**: Meaningful shift toward Sacramento
- **`d_logit_diff > +2.0`**: Strong shift, Sacramento may appear in top-5
- **`d_logit_diff > +6.0`**: Sacramento overtakes Austin (logit_diff flips sign)

### If nothing happens (d_logit_diff ≈ 0)

- Try larger `--intervene_k` (20-50 neurons)
- Try larger `--alpha` (2.0-5.0)
- Try different `--layer` (18-24 are usually most semantic)
- Check `ablation_screen_top10`: if all `d_logit_diff` are tiny (<0.1), this layer/pos may not be causal for this task

## Parameter tuning guide

| Parameter | Default | When to increase | When to decrease |
|-----------|---------|------------------|------------------|
| `screen_topk_neurons` | 2000 | If you suspect causal neurons are outside top-2000 | To speed up (but risky) |
| `screen_eval_k` | 50 | If ablation screen shows weak signal | To speed up |
| `intervene_k` | 10 | If effect is too weak | If effect is too strong/unstable |
| `alpha` | 1.0 | If effect is too weak | If logits saturate or flip unexpectedly |

## Comparison: POC3 vs POC3.1

| Feature | POC3 | POC3.1 |
|---------|------|--------|
| Logit precision | bf16 (quantized) | float32 (full precision) |
| Neuron selection | DLA proxy only | DLA + ablation screening |
| Intervention scope | 1 neuron | K neurons (default: 10) |
| Patching mode | Absolute (`a = a_src`) | Delta (`a += alpha*(a_src-a_dest)`) |
| Expected effect | +0.1 to +0.5 | +0.5 to +3.0 |
| Runtime | ~30s | ~2-5 min (ablation screening) |

## Next steps

1. Run POC3.1 on nodo207 with the commands above
2. Check `d_logit_diff` in the report
3. If weak, try `--intervene_k 20 --alpha 2.0`
4. If strong, try sweeping alpha: 0.5, 1.0, 2.0, 5.0
5. Compare ablation screen results across layers (18-24)





