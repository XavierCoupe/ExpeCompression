#!/bin/bash

#SBATCH --time=100:00:00              # job time limit
#SBATCH -J encode_dataset                  # job name
#SBATCH -o /lium/raid-a/xcoupe/out/output.out                # output file name
#SBATCH -e /lium/raid-a/xcoupe/out/error.out                 # error file name

#SBATCH --mem=50G                    # memory reservation
#SBATCH --cpus-per-task=24     # ncpu on the same node

#SBATCH --mail-type=ALL
#SBATCH --mail-user=xavier.coupe.etu@univ-lemans.fr

python3 encode_dataset.py /lium/raid-a/xcoupe/DATA/LibriSpeech /lium/scratch/xcoupe/DATA/LibriSpeech/encode hubert 24
