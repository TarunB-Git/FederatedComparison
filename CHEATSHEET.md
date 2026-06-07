# FedComparison Cheatsheet (short)

Quick commands to set up, prepare data, run experiments, and inspect results.

1) Create virtualenv and install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# For CUDA: install a matching torch build, e.g.
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

2) Prepare dataset (writes `artifacts/`)

```bash
python code/prepare.py --root . --outdir artifacts --hierarchy-taxonomy legacy8 --split-mode replay
```

3) Quick smoke end-to-end pipeline

```bash
python code/run_hier_pipeline.py --profile smoke --dataset-dir artifacts --outroot runs/hier_pipeline_smoke
```

4) Run a single paradigm

- Centralized (GRU, Prot):
```bash
python code/hier_centralized.py --dataset-dir artifacts --outdir runs/centralized_gru_Prot --arch gru --races Prot --profile smoke
```

- FedAvg:
```bash
python code/hier_fedavg.py --dataset-dir artifacts --outdir runs/fedavg_gru_Prot --arch gru --races Prot --profile smoke
```

- FedProx (with mu):
```bash
python code/hier_fedprox.py --dataset-dir artifacts --outdir runs/fedprox_gru_Prot --arch gru --races Prot --profile smoke --mu 0.01
```

- Backbone–Head (shared encoder, local heads):
```bash
python code/hier_backbone_head_race.py --dataset-dir artifacts --outroot runs/backbone_head_smoke --profile smoke --arch gru --backbone-races all
```

5) Useful flags (short):

- `--arch` / `--archs`: `gru`, `lstm`, `transformer`
- `--hidden`: encoder hidden dim (default 256)
- `--layers`: encoder depth (default 2)
- `--window`: input window size (e.g., 8)
- `--bs`: batch size
- `--epochs`: training epochs (centralized) or client epochs (federated)
- `--rounds`: federated rounds
- `--clients-per-round`: clients selected per round
- `--mu`: FedProx proximal weight
- `--device`: `cpu` or `cuda`
- `--profile`: `smoke` or `full`

6) Generate figures / parse metrics

```bash
python generate_figures.py
python parse_metrics.py runs/hier_pipeline_smoke/cross_run_results.csv
```

7) Inspect outputs

- Check run directory (`--outroot`) for `final_test.json`, checkpoints, and `cross_run_results.csv`.

8) Notes

- Use `--profile smoke` for fast iteration.
- If using GPU, ensure `torch` has CUDA support and pass `--device cuda`.
- Avoid multiple copies of `hier_models`/`pvp_raw_state_models` on `PYTHONPATH`.
