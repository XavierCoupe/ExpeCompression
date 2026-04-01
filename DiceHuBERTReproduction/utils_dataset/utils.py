import argparse
import json
import numpy as np
import os
import random
import tqdm
import torch
import torch.nn.functional as F
import torchaudio
import typing

from pathlib import Path
from torch.utils.data import Dataset

PATTERNS = ["train-*/**/*.flac", "dev-*/**/*.flac"]

def create_length_from_dataset(dataset_dir:Path, output_dir:Path):
    data = dict()
    dataset_dir = Path(dataset_dir)
    for pattern in PATTERNS :
        files_path = dataset_dir.rglob(pattern)
        for path in tqdm.tqdm(files_path):
            file = torchaudio.load(str(path))
            sample = len(file[0][0])
            path_file = str(path).split('.')[0]
            data[path_file] = sample

    json_str = json.dumps(data, indent=4)
    with open(os.path.join(output_dir, "lengths.json"), "w") as fp:
        fp.write(json_str)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create length.json file for LibriSpeech corpus.")
    parser.add_argument(
        "dataset_dir",
        metavar="dataset-dir",
        help="path to the corpus directory.",
        type=Path,
    )
    parser.add_argument(
        "output_dir",
        metavar="output-dir",
        help="path to the save file.",
        type=Path,
    )
    
    args = parser.parse_args()
    create_length_from_dataset(args.dataset_dir, args.output_dir)