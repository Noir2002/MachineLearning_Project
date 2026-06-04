#!/usr/bin/env bash
set -euo pipefail

python -m src.data_loader --validate --split test
python -m src.build_features --split test
python -m src.predict
python -m src.validate_submission results/submission.csv

