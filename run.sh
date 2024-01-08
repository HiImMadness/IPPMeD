#!/bin/sh

# SLURM options:
#SBATCH --partition=IPPMED-A40               # Choix de partition (obligatoire)
#SBATCH --ntasks=1                    # Exécuter une seule tâche
#SBATCH --nodes=1 
#SBATCH --time=1-00:00:00     # pendant 1 jours
#SBATCH --gpus=2	# avec 2 gpus

# Commandes à soumettre :
eval “$(conda shell.bash hook)”
conda activate base
python3 swinunetr_2gpu.py
