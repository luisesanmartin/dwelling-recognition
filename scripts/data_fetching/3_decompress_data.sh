#!/bin/bash
#
#SBATCH --workdir=//scratch/lsanmartin/dwelling-recognition/scripts/data_fetching
#SBATCH --job-name=decompressing_tif_files
#SBATCH --mail-user=lsanmartin@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=general
#SBATCH --mem-per-cpu=15000
#SBATCH --nodes=1
#SBATCH --ntasks=1
tar -xvf ./../../data/raw/train_tier_1.tgz -C ./../../data/raw train_tier_1/kam
