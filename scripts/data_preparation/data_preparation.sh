#!/bin/bash
#
#SBATCH --workdir=//scratch/lsanmartin/dwelling-recognition/scripts/data_preparation/
#SBATCH --job-name=preparing_labels
#SBATCH --mail-user=lsanmartin@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=general
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=data_prep.out
#SBATCH --error=data_prep.err

#python3 preparing_labels.py
python3 preparing_training_features.py
