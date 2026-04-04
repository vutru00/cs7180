# CS7180 Project

# Experiment & Script Reference

All commands should be run from the project root (`/scr/truong/code/cs7180`).

---

## Phase 1 — Causal Tracing

**`experiments/run_tracing.py`**

Identifies which layers causally mediate demographic bias. For each prompt and layer, runs three forward passes (clean, corrupted, restored) and computes the Natural Indirect Effect (NIE).

```bash
# Gender bias in UNet layers
python experiments/run_tracing.py \
    --bias-dim gender --target unet \
    --output results/nie_gender_unet.json

# Gender bias in text encoder layers
python experiments/run_tracing.py \
    --bias-dim gender --target textenc \
    --output results/nie_gender_textenc.json

# Age bias
python experiments/run_tracing.py \
    --bias-dim age --target unet \
    --output results/nie_age_unet.json
```

| Flag | Description | Default |
|------|-------------|---------|
| `--bias-dim` | `gender` or `age` | `gender` |
| `--target` | `unet`, `textenc`, or `both` | `unet` |
| `--prompts` | Prompt dataset JSON | `data/occupation_prompts.json` |
| `--n-images` | Seeds per prompt | `5` |
| `--output` | Output JSON path (required) | — |

**Output:** JSON mapping `{prompt → {layer_name → NIE value}}`.

---

## Extract Top Causal Layers

**`scripts/extract_top_layers.py`**

Ranks layers by absolute NIE from Phase 1 results and saves the top-k as a JSON list for use in Phase 2 and 3.

```bash
python scripts/extract_top_layers.py \
    --inputs results/nie_gender_unet.json results/nie_gender_textenc.json \
    --top-k 10 --print-all \
    --output results/top_causal_layers.json
```

| Flag | Description | Default |
|------|-------------|---------|
| `--inputs` | One or more NIE JSON files (required) | — |
| `--top-k` | Number of layers to keep | `10` |
| `--print-all` | Print full ranked list | off |
| `--output` | Output JSON path | `results/top_causal_layers.json` |

---

## Phase 2 — Dynamic Mediation

**`experiments/run_dynamic.py`**

Same as Phase 1 but restricts the restoration to a specific timestep window, revealing *when* during denoising each layer exerts its causal influence.

```bash
python experiments/run_dynamic.py \
    --window early --layers results/top_causal_layers.json \
    --output results/dynamic_early.json

python experiments/run_dynamic.py \
    --window mid --layers results/top_causal_layers.json \
    --output results/dynamic_mid.json

python experiments/run_dynamic.py \
    --window late --layers results/top_causal_layers.json \
    --output results/dynamic_late.json
```

| Flag | Description | Default |
|------|-------------|---------|
| `--window` | `early` (999→700), `mid` (700→300), `late` (300→0) — required | — |
| `--layers` | JSON file with layer names to probe (required) | — |
| Other flags | Same as `run_tracing.py` | — |

---

## Phase 3 — Interventions

**`experiments/run_intervention.py`**

Generates images with a debiasing intervention active, scores them with MiVOLO, and compares against baseline. Supports `--visualize` to save per-prompt image grids and a summary chart.

### Methods

| Method | Description | Requires |
|--------|-------------|----------|
| `hard_block` | Zero out causal layer outputs | `--causal-layers` |
| `soft_steer` | Subtract a gender steering vector | `--causal-layers` |
| `patching` | Swap causal layers with de-biased activations | `--causal-layers` |
| `pca_projection` | Project out the PCA gender subspace | `--causal-layers` |
| `prompt_aug` | Prepend "a person who is a" to the prompt | — |
| `random_block` | Zero out random non-causal layers (control) | — |

### Examples

```bash
# Hard block on UNet causal layers
python experiments/run_intervention.py \
    --method hard_block --target unet \
    --causal-layers results/top_causal_layers.json \
    --output results/intervention_hard_block.csv --visualize

# Patching on text encoder layers
python experiments/run_intervention.py \
    --method patching --target textenc \
    --causal-layers results/top_causal_layers.json \
    --output results/intervention_patching_te.csv --visualize

# PCA projection (contextual pairs, EOS token)
python experiments/run_intervention.py \
    --method pca_projection --target textenc \
    --causal-layers results/top_causal_layers.json \
    --pair-type contextual --token-position eos --n-components 1 \
    --output results/intervention_pca.csv --visualize

# Restrict intervention to early timesteps only
python experiments/run_intervention.py \
    --method hard_block --target unet --window early \
    --causal-layers results/top_causal_layers.json \
    --output results/intervention_block_early.csv --visualize

# Prompt augmentation baseline (no causal layers needed)
python experiments/run_intervention.py \
    --method prompt_aug \
    --output results/intervention_prompt_aug.csv --visualize
```

| Flag | Description | Default |
|------|-------------|---------|
| `--method` | Intervention method (required) | — |
| `--target` | `unet` or `textenc` | `unet` |
| `--causal-layers` | JSON file with layer names | — |
| `--window` | `early`/`mid`/`late` (UNet only) | all timesteps |
| `--n-images` | Seeds per prompt | `5` |
| `--output` | Output CSV path (required) | — |
| `--visualize` | Save image grids + summary chart | off |
| `--viz-dir` | Visualization directory | `results/<method>/` |

PCA-specific flags (only for `--method pca_projection`):

| Flag | Description | Default |
|------|-------------|---------|
| `--gender-pairs` | Gender contrastive pairs JSON | `data/gender_pairs.json` |
| `--pair-type` | `definitional` or `contextual` | `contextual` |
| `--token-position` | `eos`, `last_subject`, or `occupation` | `eos` |
| `--n-components` | PCA components for gender subspace | `1` |

### Output

- **CSV** with columns: `condition, method, prompt, image_idx, gender_score, age, clip_score, detected`. Each prompt has both `baseline` and `intervention` rows.
- **`--visualize`** creates:
  - `<viz-dir>/images/<occupation>/` — individual PNGs
  - `<viz-dir>/grids/<occupation>.png` — side-by-side (original vs intervention)
  - `<viz-dir>/summary.png` — bar chart of male probability per prompt

---

## Validate PCA Gender Direction

**`scripts/validate_pca_gender.py`**

Computes the PCA gender direction for every text encoder layer and validates it with two checks: (1) separation accuracy on held-out gendered prompts, (2) occupation-gender entanglement.

```bash
# Train on contextual pairs, validate with definitional (held-out)
python scripts/validate_pca_gender.py \
    --train-pairs contextual --holdout-pairs definitional \
    --token-position eos --n-components 1 \
    --output results/pca_validation.png

# Only validate specific layers
python scripts/validate_pca_gender.py \
    --layers results/top_causal_layers.json \
    --output results/pca_validation_causal.png
```

| Flag | Description | Default |
|------|-------------|---------|
| `--train-pairs` | `definitional` or `contextual` | `contextual` |
| `--holdout-pairs` | `definitional` or `contextual` | `definitional` |
| `--token-position` | `eos`, `last_subject`, or `occupation` | `eos` |
| `--n-components` | PCA components | `1` |
| `--layers` | JSON file with specific layers (default: all 24) | — |
| `--output` | Output PNG path | `results/pca_validation.png` |

**Output:** PNG with 4 panels (separation accuracy, entanglement ratio, PCA dominance, singular value spectrum) + JSON with full per-layer results.

---

## Activation Swap Experiment

**`scripts/experiment_swap_activations.py`**

Diagnostic experiment for "nurse": generates with "male nurse" and "female nurse", records every layer's activations, then swaps one layer at a time from the opposite gender's run. Reports whether swapping each layer flips the output gender.

```bash
# Text encoder only (fast, ~2 min)
python scripts/experiment_swap_activations.py --target textenc --n-images 5

# UNet only
python scripts/experiment_swap_activations.py --target unet --n-images 3

# Both
python scripts/experiment_swap_activations.py --target both --n-images 5
```

| Flag | Description | Default |
|------|-------------|---------|
| `--target` | `textenc`, `unet`, or `both` | `both` |
| `--n-images` | Seeds to average over | `5` |
| `--base-seed` | Starting seed | `42` |
| `--output-dir` | Output directory | `results/swap_experiment/` |

**Output:** `swap_results.json` + bar chart PNGs showing male probability after swap per layer.

---

## Typical Workflow

```
Phase 1: run_tracing.py          →  NIE JSON
              ↓
         extract_top_layers.py    →  top_causal_layers.json
              ↓
Phase 2: run_dynamic.py           →  timestep-resolved NIE JSON
              ↓
Phase 3: run_intervention.py      →  CSV + visualizations

Diagnostics (run independently):
  validate_pca_gender.py          →  PCA validation plots
  experiment_swap_activations.py  →  layer-level gender mediation
```
