#!/bin/bash
# Reports 07 and 08 from the converted days (run after itch_days.sbatch finishes).
#SBATCH --job-name=markout-many-days
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/many_days_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
python -m markout.lob.multiday
