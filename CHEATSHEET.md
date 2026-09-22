# FedComparison command reference

These commands use flags accepted by the current scripts. The single-run examples reproduce the saved paper protocol for one GRU configuration and are computationally expensive.

## Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU experiments, replace the CPU PyTorch wheel with the build matching the local CUDA runtime.

## Prepare the dataset

```bash
python code/prepare.py \
  --root . \
  --outdir artifacts \
  --hierarchy-taxonomy legacy8 \
  --split-mode replay
```

See [`DATA.md`](DATA.md) for the required source archives and directory layout.

## Run an end-to-end smoke test

```bash
python code/prepare.py \
  --root . \
  --outdir artifacts_smoke \
  --hierarchy-taxonomy legacy8 \
  --split-mode replay \
  --smoke

python code/run_hier_pipeline.py \
  --profile smoke \
  --dataset-dir artifacts_smoke \
  --outroot runs/hier_pipeline_smoke \
  --workers 0
```

## Run one saved-protocol configuration

Centralized GRU on Protoss:

```bash
python code/hier_centralized.py \
  --dataset-dir artifacts \
  --outdir runs/centralized_gru_prot \
  --model-name gru \
  --race Prot \
  --window 8 \
  --hidden 256 \
  --layers 2 \
  --dropout 0.2 \
  --epochs 20 \
  --bs 512 \
  --lr 0.001 \
  --weight-decay 1e-5 \
  --early-stop-patience 100 \
  --workers 8 \
  --seed 123 \
  --device cuda \
  --resume
```

FedAvg GRU on Protoss:

```bash
python code/hier_fedavg.py \
  --dataset-dir artifacts \
  --outdir runs/fedavg_gru_prot \
  --model-name gru \
  --race Prot \
  --window 8 \
  --hidden 256 \
  --layers 2 \
  --dropout 0.2 \
  --rounds 50 \
  --clients-per-round 25 \
  --local-epochs 1 \
  --local-bs 512 \
  --local-lr 0.001 \
  --local-weight-decay 1e-5 \
  --max-client-samples 2000 \
  --max-local-batches 75 \
  --eval-bs 256 \
  --workers 0 \
  --seed 123 \
  --device cuda \
  --resume
```

FedProx GRU on Protoss:

```bash
python code/hier_fedprox.py \
  --dataset-dir artifacts \
  --outdir runs/fedprox_gru_prot \
  --model-name gru \
  --race Prot \
  --mu 0.01 \
  --window 8 \
  --hidden 256 \
  --layers 2 \
  --dropout 0.2 \
  --rounds 50 \
  --clients-per-round 25 \
  --local-epochs 1 \
  --local-bs 512 \
  --local-lr 0.001 \
  --local-weight-decay 1e-5 \
  --max-client-samples 2000 \
  --max-local-batches 75 \
  --eval-bs 256 \
  --workers 0 \
  --seed 123 \
  --device cuda \
  --resume
```

Backbone-head GRU trained jointly across all races:

```bash
python code/hier_backbone_head_race.py \
  --dataset-dir artifacts \
  --outdir runs/backbone_head_gru \
  --model-name gru \
  --race all \
  --window 8 \
  --hidden 256 \
  --layers 2 \
  --dropout 0.2 \
  --rounds 150 \
  --clients-per-round 25 \
  --local-epochs 1 \
  --local-bs 512 \
  --local-lr 0.001 \
  --local-weight-decay 0 \
  --max-client-samples 0 \
  --max-local-batches 75 \
  --eval-bs 256 \
  --round-val-clients 50 \
  --workers 0 \
  --seed 123 \
  --device cuda \
  --resume
```

Replace `gru` with `lstm` or `transformer`, and choose `Prot`, `Terr`, or `Zerg` for race-specific runs. Backbone-head paper runs use `--race all`.

## Generate figures and run tests

```bash
python scripts/generate_readme_figures.py
python -m unittest discover -s tests -v
```

The full protocol, including the backbone-head exceptions, is recorded in [`results/paper_protocol.json`](results/paper_protocol.json).
