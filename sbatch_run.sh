#!/usr/bin/env bash
#SBATCH --job-name=Prob-CBR
#SBATCH --output=logs/obl-per-relation-%A_%a.out
#SBATCH --partition=longq
#SBATCH --time=07-00:00:00
#SBATCH --mem=40G
#SBATCH --array=0-49
# Set to scratch/work since server syncing will occur from here
# Ensure sufficient space else runs crash without error message

# Flag --count specifies number of hyperparam settings (runs) tried per job
# If SBATCH --array flag is say 0-7 (8 jobs) then total (8 x count)
# hyperparam settings will be tried
#source ~/.bashrc
#conda activate obl
#wandb agent rajarshd/Open-BIo-Link-src_prob_cbr/9neli3pq

#source /mnt/nfs/work1/mccallum/rajarshi/Dropbox/research/wandb_running_code/Open-BIo-Link/wandb_settings.sh
#export WANDB_DIR="./wandb_dir/"
#export TMPDIR="./tmp_dir/"
#export PYTHONUNBUFFERED=1
#pwd

#conda activate pyenv
cd src
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..
# https://wandb.ai/rajarshd/Open-BIo-Link-src_prob_cbr/sweeps/pba7tork
wandb agent rajarshd/Open-BIo-Link-src_prob_cbr/pba7tork