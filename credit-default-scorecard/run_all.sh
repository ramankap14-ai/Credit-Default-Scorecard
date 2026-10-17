#!/usr/bin/env bash
set -e
python src/01_clean_load.py
python src/02_statistics.py
python src/03_scorecard.py
python src/04_run_sql.py
