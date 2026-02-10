# Predictor-Driven Diffusion

Predictor-Driven Diffusion is a scale-aware diffusion framework that unifies causal prediction and spatiotemporal generation via renormalization-inspired spatial coarse-graining and a path-integral formulation.

## Setup

1. Install `Dev Container` extension in your VSCode.
2. `Rebuild and Reopen in Container` on your VSCode.

## Experiments: Lorenz96 system

1. Run [`make_data_lorenz96.py`](./scripts/make_data_lorenz96.py) to make lorenz96 data: `python3 make_data_lorenz96.py`
2. Run [`train_pdd_lorenz96.py`](./scripts/train_pdd_lorenz96.py) to train a diffusion model: `python3 train_pdd_lorenz96.py --device cuda:0 --config_path /workspace/configs/lorenz96_unet.yml`
3. Analyze the results by running [lorenz96.ipynb](./notebooks/lorenz96.ipynb)

## Experiments: Kolmogorov-flow system

1. Run [`make_data_kolmogorov_flow.py`](./scripts/make_data_kolmogorov_flow.py) to make kolmogorov-flow data: `
