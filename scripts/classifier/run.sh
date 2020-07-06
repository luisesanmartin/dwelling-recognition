#!/bin/bash
#
#SBATCH --workdir=//scratch/lsanmartin/dwelling-recognition/scripts/classifier
#SBATCH --job-name=dwelling_train
#SBATCH --mail-user=lsanmartin@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=sbatch_outputs/train_seg.out
#SBATCH --error=sbatch_outputs/train_seg.err
#SBATCH --partition=titan
#SBATCH --mem-per-cpu=10000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

python3 classifier.py