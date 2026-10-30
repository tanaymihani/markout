#!/bin/bash
# Convert the NASDAQ ITCH sample days on Northeastern's Explorer cluster, one Slurm array task per day.
#
#   sbatch scripts/explorer/itch_days.sbatch          # ten tasks, one per day in scripts/download_itch.py
#   sbatch --dependency=afterok:<jobid> scripts/explorer/many_days.sbatch   # then reports 07 and 08
#
# Each task downloads its day (3.5-6 GB) to scratch, converts the ten stocks of report 07 and
# keeps only the LOBSTER-format Parquet files (a few hundred MB). A laptop does the same in
# about 3 minutes a day; the cluster is for many more days or tickers. Adjust the partition,
# the time limit and the environment lines to your account.
#SBATCH --job-name=markout-itch
#SBATCH --partition=short
#SBATCH --array=0-9
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/itch_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# environment: a venv with the package installed (pip install -e ".[cpp]" and make cpp)
source .venv/bin/activate

# keep the raw downloads on scratch, not in the home directory (data/raw/itch points there)
SCRATCH_ITCH="/scratch/${USER}/itch"
mkdir -p "${SCRATCH_ITCH}" data/raw
# array tasks start together, so tolerate another task having made the link first
[ -e data/raw/itch ] || ln -s "${SCRATCH_ITCH}" data/raw/itch 2>/dev/null || [ -L data/raw/itch ]

DAY=$(python -c "import runpy; print(runpy.run_path('scripts/download_itch.py')['DAYS'][${SLURM_ARRAY_TASK_ID}])")
python scripts/download_itch.py --days "${DAY%%.*}"
