#!/usr/bin/env bash
set -euo pipefail

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON:-python}"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

"$PYTHON_BIN" -m src.data_loader --validate --split test
"$PYTHON_BIN" -m src.build_features --split test
"$PYTHON_BIN" -m src.predict
"$PYTHON_BIN" -m src.validate_submission results/submission.csv
