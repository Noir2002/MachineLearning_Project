#!/usr/bin/env bash
set -euo pipefail

python -m compileall src
pytest -q
python -m src.data_loader --validate --split train
python -m src.build_features --split train
python -m src.train_models
python -m src.clustering
python -m src.visualize
python -m src.make_report

