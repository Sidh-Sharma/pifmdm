# pifmdm — Physics-Informed Flow & Diffusion Models

Benchmarking framework for evaluating how well probabilistic diffusion and flow-matching models preserve **physical structure** (conservation laws, semigroup consistency, transport geometry) when applied to PDE-governed data.

Four models are wrapped behind a unified adapter interface so they can be trained on the **same data**, evaluated with the **same diagnostics**, and compared fairly.

## Models

| Model | Type | Paper idea | Code |
|-------|------|------------|------|
| **CSDI** | Score-based diffusion | Conditional imputation via score matching | `models/csdi/` |
| **CFMI** | Conditional flow matching | ODE-based imputation with flow matching | `models/cfmi/` |
| **PIDM** | Physics-informed diffusion | x₀-prediction + optional PDE residual loss | `models/pidm/` |
| **TMDM** | Transformer-modulated diffusion | NS-Transformer VAE + diffusion forecasting | `models/tmdm/` |

Each model is trained using **its own native training code** (optimizer, scheduler, EMA, early stopping, etc.). The adapters are thin wrappers that handle data format conversion in and out.

## Diagnostics

| Diagnostic | Hypothesis | What it measures |
|------------|-----------|-----------------|
| **Semigroup consistency** | H1, H2 | Does Φ(t₁+t₂) ≈ Φ(t₂) ∘ Φ(t₁)? |
| **Conservation error** | H3 | Are integral quantities preserved across samples? |
| **Error vs. horizon** | H1, H2 | How does RMSE/MAE grow with prediction horizon? |
| **Error vs. sparsity** | H1, H2 | How does accuracy degrade with missing data? |
| **Wasserstein distance** | H1 | Does the model blur the true transport structure? |

## Project structure

```
adapters/            Unified adapter interface + per-model wrappers
  base.py            BaseAdapter (prepare_native_loaders, native_train, native_evaluate, predict)
  csdi_adapter.py    CSDI wrapper → delegates to utils.train() / utils.evaluate()
  cfmi_adapter.py    CFMI wrapper → delegates to PyTorch Lightning Trainer.fit()
  pidm_adapter.py    PIDM wrapper → replicates main.py (Adam, EMA, iteration-based)
  tmdm_adapter.py    TMDM wrapper → subclasses Exp_Main with overridden data loading
configs/experiment/  YAML configs (one per model × dataset combination)
datasets/
  base.py            UnifiedDataset, collate, dataloader utilities
  synthetic/         Linear advection data + loader
diagnostics/         Semigroup, conservation, horizon, sparsity, Wasserstein
models/              Original model repos (CSDI, CFMI, PIDM, TMDM)
scripts/
  run_experiment.py  End-to-end pipeline: data → train → eval → diagnostics
  evaluate.py        Standalone eval from a saved checkpoint
results/             Checkpoints, metrics, and diagnostic outputs
```

## Quick start

```bash
# Train CSDI on advection data, auto-select GPU
python scripts/run_experiment.py \
  --model csdi \
  --config configs/experiment/csdi_advection.yaml \
  --device auto

# Same for the other three models
python scripts/run_experiment.py --model cfmi --config configs/experiment/cfmi_advection.yaml --device auto
python scripts/run_experiment.py --model pidm --config configs/experiment/pidm_advection.yaml --device auto
python scripts/run_experiment.py --model tmdm --config configs/experiment/tmdm_advection.yaml --device auto
```

`--device auto` picks CUDA → MPS → CPU in that order.

## Evaluate a saved checkpoint

```bash
python scripts/evaluate.py \
  --model csdi \
  --config configs/experiment/csdi_advection.yaml \
  --checkpoint results/checkpoints/csdi_advection/best_model.pt \
  --n_samples 50 \
  --device auto \
  --run_diagnostics
```

## Pipeline flow

```
UnifiedDataset                        (B, L, K) time-first tensors
       │
adapter.prepare_native_loaders()      wraps into model-specific format
       │
adapter.native_train(loaders)         runs each model's own training loop
       │
adapter.native_evaluate(loaders)      runs each model's own sampling + metrics
       │
adapter.predict(batch)                unified interface for diagnostics
       │
diagnostics.*                         semigroup, conservation, horizon, wasserstein
       │
results/metrics/                      JSON metrics + .npy arrays
```

## Config anatomy

Each YAML config has five sections:

```yaml
data:           # dataset type, path, subsampling, masking
csdi:           # (or cfmi/pidm/tmdm) model-specific hyperparameters
training:       # save_dir, epochs/iterations (model-dependent)
evaluation:     # n_samples, save_dir
diagnostics:    # which diagnostics to run and their parameters
```

`target_dim` and `seq_len` are inferred automatically from the data.
