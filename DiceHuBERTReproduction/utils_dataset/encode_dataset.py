import argparse
import json
import multiprocessing
import numpy as np
import os
import pathlib
import random
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import typing

from multiprocessing import Pool
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2Model, Wav2Vec2FeatureExtractor,
    HubertModel,
    ASTModel, ASTFeatureExtractor,
    AutoModel, AutoFeatureExtractor
)






PATTERNS = ["train-*/**/*.flac", "dev-*/**/*.flac"]

    
def init_pool(output_dir:Path, encoder_type:str, manager:multiprocessing.Manager):
    global encoder
    global save_dir

    
    
    ####
    # Load Encoder 
    ####
    
    if encoder_type.upper() == "HUBERT":
        model_name = "facebook/hubert-base-ls960"
        encoder = HubertModel.from_pretrained(model_name)
        freeze_model(encoder)
    else:
        raise Exception(f"!!! No audio encoder can be found with {encoder_type=} !!!")

    if count_parameters(encoder):
        raise Exception(f"!!! Encoder not freeze !!!")

    ####
    # Init save dir
    ####

    save_dir = output_dir
    

def encode_wav(path):

    all_path = path.split("LibriSpeech")[1]
    split_path = all_path.split("/")
    save_path = "/".join(split_path[: len(split_path)-1])[1:]
    
    file_name = all_path.split("/")[-1]
    
    path_pathlib = Path(save_path)
    absolute_path = Path(pathlib.PurePath(save_dir, path_pathlib))

    # Verification si le fichier a déjà été traité
    exist_file = Path(str(absolute_path) + "/" + file_name + ".npy")
    if exist_file.is_file():
        return 
    
    wav, _ = torchaudio.load(str(path) + ".flac")
    
    ####
    # Encode wav
    ####
    
    outputs = encoder(
        wav, 
    )
    
    ####
    # Save encode wav
    ####
    
    
    absolute_path.mkdir(parents=True, exist_ok=True)

    data = outputs.last_hidden_state.numpy()
    np.save(str(absolute_path) + "/" + file_name, data)
    
    
def freeze_model(model, eval_mode=True):
    """
    Completely freeze a model, including BatchNorm statistics.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to freeze
    eval_mode : bool, default=True
        Whether to put the model in evaluation mode
        
    Returns:
    --------
    model : torch.nn.Module
        The frozen model
    """
    # Step 1: Set all parameters to not require gradients
    for param in model.parameters():
        param.requires_grad = False
    
    # Step 2: Set model to evaluation mode to freeze BatchNorm statistics
    if eval_mode:
        model.eval()
    
    # Step 3: Override the BatchNorm layers to ensure they don't update statistics
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False  # Don't track running stats
            module.running_mean = module.running_mean.detach()  # Detach running mean
            module.running_var = module.running_var.detach()    # Detach running variance
    
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create length.json file for LibriSpeech corpus.")

    parser.add_argument(
        "lengths_dir",
        metavar="lengths-dir",
        help="path to the lengths.json file.",
        type=Path,
    )
    
    parser.add_argument(
        "output_dir",
        metavar="output-dir",
        help="path to the save file.",
        type=Path,
    )

    parser.add_argument(
        "encoder",
        metavar="encoder",
        help="encoder to use. wavlm or hubert",
        type=str,
    )

    parser.add_argument(
        "nb_process",
        metavar="nb-process",
        help="Number of process to launch",
        type=int,
    )
    
    args = parser.parse_args()
    
    # for path in tqdm.tqdm(lengths_dict) :
    manager = multiprocessing.Manager()
    
    ####
    # Load lengths file
    ####

    with open(args.lengths_dir / "lengths.json") as file:
        lengths_dict = manager.dict(json.load(file))
        
    pool = Pool(initializer=init_pool, initargs=(args.output_dir, args.encoder, manager), processes=args.nb_process)
    pool.map(encode_wav, lengths_dict.keys())

        














