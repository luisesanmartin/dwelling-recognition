#!/bin/bash
#
#SBATCH --workdir=//scratch/lsanmartin/dwelling-recognition/notebooks/data_preparation/
#SBATCH --job-name=preparing_labels
#SBATCH --mail-user=lsanmartin@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=general
#SBATCH --mem-per-cpu=15000
#SBATCH --nodes=1
#SBATCH --ntasks=1
python3 preparing_labels.py
