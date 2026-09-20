# Predictor-Driven Diffusion

Predictor-Driven Diffusion (PDD) is a scale-aware diffusion framework that unifies causal prediction and spatiotemporal generation via renormalization-inspired spatial coarse-graining and a path-integral formulation.

Score-Augmented Predictor-Driven Diffusion (SA-PDD) adds a second pointwise output head that estimates the forward-process noise and trains with `score_loss + drift_loss_weight * drift_loss`.

## Setup

### Dev Container

1. Install `Dev Container` extension into VSCode.
2. `Rebuild and Reopen in Container` on VSCode.
3. Check if the extensions specified in [`devcontainer.json`](./.devcontainer/devcontainer.json) are installed in the container.

### Virtual Environment

1. Install Poetry.
2. Build a virtual environment and register a Jupyter kernel with `PYTHONPATH` set to the repository root (run from the repository root, so `$PWD` resolves correctly):
    ```sh
    poetry env use python3.11
    poetry install --no-root
    poetry run python -m ipykernel install --user --name predictor-driven-diffusion \
      --display-name "Python (predictor-driven-diffusion)" --env PYTHONPATH "$PWD"
    ```
3. Select the kernel `Python (predictor-driven-diffusion)` when you run a notebook.

From the repository root, scripts can be run as
`PYTHONPATH="$PWD" poetry run python3 scripts/train_pdd_lorenz96.py --device cuda:0 --config_path configs/lorenz96_pdd.yml`.
In the Dev Container, the repository root is `/workspace`.

## Experiments: Lorenz96 system (PDD)

1. `Reopen in Container`
2. Run [`make_data_lorenz96.py`](./scripts/make_data_lorenz96.py) to make lorenz96 data: `$ python3 scripts/make_data_lorenz96.py`
3. Run [`train_pdd_lorenz96.py`](./scripts/train_pdd_lorenz96.py) to train a diffusion model: `$ python3 scripts/train_pdd_lorenz96.py --device cuda:0 --config_path configs/lorenz96_pdd.yml`
4. Analyze the results by running [lorenz96.ipynb](./notebooks/lorenz96.ipynb)

## Experiments: Kolmogorov-flow system (PDD)

1. `Reopen in Container`
2. Run [`make_data_kolmogorov_flow.py`](./scripts/make_data_kolmogorov_flow.py) to make kolmogorov-flow data: `$ python3 scripts/make_data_kolmogorov_flow.py`
3. Run [`train_pdd_kolmogorov_flow.py`](./scripts/train_pdd_kolmogorov_flow.py) to train a diffusion model: `$ python3 scripts/train_pdd_kolmogorov_flow.py --device cuda:0 --config_path configs/kolmogorov_flow_pdd.yml`
4. Analyze the results by running [kolmogorov_flow.ipynb](./notebooks/kolmogorov_flow.ipynb)

## Experiments: Lorenz96 system (SA-PDD)

1. `Reopen in Container`
2. Run [`make_data_lorenz96.py`](./scripts/make_data_lorenz96.py) to make lorenz96 data, or skip this step if the data already exist: `$ python3 scripts/make_data_lorenz96.py`
3. Run [`train_sa_pdd_lorenz96.py`](./scripts/train_sa_pdd_lorenz96.py) to train a diffusion model: `$ python3 scripts/train_sa_pdd_lorenz96.py --device cuda:0 --config_path configs/lorenz96_sa_pdd.yml`
4. Analyze the results by running [lorenz96_sa_pdd.ipynb](./notebooks/lorenz96_sa_pdd.ipynb)

## Experiments: Kolmogorov-flow system (SA-PDD)

1. `Reopen in Container`
2. Run [`make_data_kolmogorov_flow.py`](./scripts/make_data_kolmogorov_flow.py) to make kolmogorov-flow data, or skip this step if the data already exist: `$ python3 scripts/make_data_kolmogorov_flow.py`
3. Run [`train_sa_pdd_kolmogorov_flow.py`](./scripts/train_sa_pdd_kolmogorov_flow.py) to train a diffusion model: `$ python3 scripts/train_sa_pdd_kolmogorov_flow.py --device cuda:0 --config_path configs/kolmogorov_flow_sa_pdd.yml`
4. Analyze the results by running [kolmogorov_flow_sa_pdd.ipynb](./notebooks/kolmogorov_flow_sa_pdd.ipynb)
