# Predictor-Driven Diffusion

Predictor-Driven Diffusion is a scale-aware diffusion framework that unifies causal prediction and spatiotemporal generation via renormalization-inspired spatial coarse-graining and a path-integral formulation.

## Setup

1. Install `Dev Container` extension into VSCode.
2. `Rebuild and Reopen in Container` on VSCode.
3. Check if the extensions specified in [`devcontainer.json`](./.devcontainer/devcontainer.json) are installed in the container.

## Experiments: Lorenz96 system

1. `Reopen in Container`
2. Run [`make_data_lorenz96.py`](./scripts/make_data_lorenz96.py) to make lorenz96 data: `$ python3 make_data_lorenz96.py`
3. Run [`train_pdd_lorenz96.py`](./scripts/train_pdd_lorenz96.py) to train a diffusion model: `$ python3 train_pdd_lorenz96.py --device cuda:0 --config_path /workspace/configs/lorenz96_unet.yml`
4. Analyze the results by running [lorenz96.ipynb](./notebooks/lorenz96.ipynb)

## Experiments: Kolmogorov-flow system

1. `Reopen in Container`
2. Run [`make_data_kolmogorov_flow.py`](./scripts/make_data_kolmogorov_flow.py) to make kolmogorov-flow data: `$ python3 make_data_kolmogorov_flow.py`
3. Run [`train_pdd_kolmogorov_flow.py`](./scripts/train_pdd_kolmogorov_flow.py) to train a diffusion model: `$ python3 train_pdd_kolmogorov_flow.py --device cuda:0 --config_path /workspace/configs/kolmogorov_flow_unet.yml`
4. Analyze the results by running [kolmogorov_flow.ipynb](./notebooks/kolmogorov_flow.ipynb)
