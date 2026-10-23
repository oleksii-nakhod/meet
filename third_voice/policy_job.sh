#!/bin/bash
#SBATCH -p gpunodes
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 00:30:00
#SBATCH -o /w/20251/oleksii/code/boson2026/builds/third_voice/out/slurm-%j.out
set -e
cd /w/20251/oleksii/code/boson2026/builds/third_voice
python3 -m venv /tmp/tv-venv-$SLURM_JOB_ID
V=/tmp/tv-venv-$SLURM_JOB_ID/bin
$V/pip install -q --index-url https://download.pytorch.org/whl/cu128 torch
$V/pip install -q "transformers==5.5.3" peft accelerate
export HF_HOME=/tmp/hf-$SLURM_JOB_ID
$V/python -u policy_gpu.py
rm -rf /tmp/tv-venv-$SLURM_JOB_ID /tmp/hf-$SLURM_JOB_ID
