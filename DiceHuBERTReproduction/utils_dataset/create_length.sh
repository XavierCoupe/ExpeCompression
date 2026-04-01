#!/bin/bash

#SBATCH --time=5:00:00              # job time limit
#SBATCH -J librispeech                  # job name
#SBATCH -o /lium/raid-a/xcoupe/out/output.out                # output file name
#SBATCH -e /lium/raid-a/xcoupe/out/error.out                 # error file name

#SBATCH --mem=10G                    # memory reservation
#SBATCH --cpus-per-task=5          # ncpu on the same node

#SBATCH --mail-type=ALL
#SBATCH --mail-user=xavier.coupe.etu@univ-lemans.fr

python3 utils.py /lium/corpus/base/LibriSpeech /lium/raid-a/xcoupe/DATA/LibriSpeech