#!/usr/bin/env bash
set -euo pipefail

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON:-python}"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

"$PYTHON_BIN" -m compileall src
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" -m src.data_loader --validate --split train
"$PYTHON_BIN" -m src.build_features --split train
"$PYTHON_BIN" -m src.train_models
"$PYTHON_BIN" -m src.clustering
"$PYTHON_BIN" -m src.visualize
"$PYTHON_BIN" -m src.make_report
