#!/bin/bash

#SBATCH --time=10:00:00              # job time limit
#SBATCH -J train_dicehubert                  # job name
#SBATCH -o /lium/raid-a/xcoupe/out/output.out                # output file name
#SBATCH -e /lium/raid-a/xcoupe/out/error.out                 # error file name

#SBATCH --mem=50G                    # memory reservation
#SBATCH --cpus-per-task=5          # ncpu on the same node
#SBATCH -p gpu --gres gpu:rtx8000:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=xavier.coupe.etu@univ-lemans.fr

python3 -u train.py /lium/corpus/base/LibriSpeech/ ./logs/