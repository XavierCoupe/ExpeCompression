import json
import numpy as np
import pickle
import random
import tqdm
import torch
import torch.nn.functional as F
import torchaudio

from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import Dataset



class AcousticUnitsDataset(Dataset):
    def __init__(
        self,
        root: Path,
        root_length: Path,
        root_discrete: Path,
        model_path: Path = Path("/lium/scratch/xcoupe/DATA/LibriSpeech/kmeans_model/"),
        sample_rate: int = 16000,
        label_rate: int = 50,
        min_samples: int = 32000,
        max_samples: int = 250000,
        nb_centroid: int = 500,
        batch_size: int = 256,
        seed: int = 20,
        nb_dim: int = 200, # Nombre de dimension à garder pour le kmeans
        force_train_kmeans: bool = False,
        train: bool = True,
    ):
        self.nb_dim = nb_dim
        self.wavs_dir = root
        self.units_dir = root_discrete

        self.kmeans = MiniBatchKMeans(n_clusters=nb_centroid, 
                                               init="k-means++",
                                               random_state=seed,
                                               batch_size=256,
                                               max_iter=100,
                                               n_init="auto")

        with open(root_length / "lengths.json") as file:
            self.lengths = json.load(file)

        # Si le modèle est déjà existant, on passe l'apprentissage et charge le modèle enregistré
        model_name = f"kmeans{nb_centroid}_seed{seed}"
        exist_file = Path(str(model_path) + "/" + model_name + ".pkl")
        if exist_file.is_file() and not force_train_kmeans:
            print("[MODEL] Chargement du modèle...")
            with open(str(model_path) + "/" + model_name + ".pkl", 'rb') as f:
                self.kmeans = pickle.load(f)
        else :
            # Sélection de 10% du dataset pour fit le kmeans
            if force_train_kmeans :
                print("[MODEL] Réapprentissage du modèle...")
            print("[MODEL] Préparation des données pour apprentissage...")
            kmeans_pattern = "train-clean-100/**/*.npy"
    
            kmeans_metadata = (
                (path, path.relative_to(self.units_dir).with_suffix("").as_posix())
                for path in self.units_dir.rglob(kmeans_pattern)
            )

            kmeans_corpus = list()
            for path, key in tqdm.tqdm(list(kmeans_metadata)):
                wav = np.load(str(path))
                
                # Récupère les éléments qui ont au minimun nb_dim trames
                wav = np.squeeze(wav, axis=0)

                # Tronque les éléments qui ont trop d'éléments
                if wav.shape[0] > nb_dim:
                    idx = random.randint(0, wav.shape[0] - nb_dim)
                    kmeans_corpus.append(wav[idx:idx+nb_dim,:])
                                     
            print("[MODEL] Reformatage des données....")
            kmeans_corpus = np.array(kmeans_corpus)

            
            #MiniBatchKmeans ne prend que des arrays qui ont 2 ou moins de dimensions
            kmeans_corpus = kmeans_corpus.reshape((kmeans_corpus.shape[0]*kmeans_corpus.shape[1], kmeans_corpus.shape[2]))                                       
            print("[MODEL] Apprentissage du modèle....")
            self.kmeans.fit(kmeans_corpus)

            print("[MODEL] Sauvegarde du modèle....")
            with open(str(model_path) + "/" + model_name + ".pkl",'wb') as f:
                pickle.dump(self.kmeans,f)
        
        pattern = "train-*/**/*.flac" if train else "dev-*/**/*.flac"
        metadata = (
            (path, path.relative_to(self.wavs_dir).with_suffix("").as_posix())
            for path in self.wavs_dir.rglob(pattern)
        )
        
        metadata = ((path, key) for path, key in metadata if key in self.lengths)

        self.metadata = [
            path for path in self.lengths if self.lengths[path] > min_samples
        ]

        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.train = train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        wav_path = self.metadata[index]
        wav_path = Path(wav_path)
        units_path = self.units_dir / wav_path.relative_to(self.wavs_dir)

        wav, _ = torchaudio.load(str(wav_path) + ".flac")
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))

        codes = np.load(units_path.with_suffix(".npy"))
        codes = np.squeeze(codes, axis=0)
        
        # prediction des centroids
        if codes.shape[0] > self.nb_dim:
            idx = random.randint(0, codes.shape[0] - self.nb_dim)
            codes = codes[idx:idx+self.nb_dim,:]

        else :
            # Ajout de padding
            target_size = (self.nb_dim,codes.shape[1])


            padded = np.zeros(target_size, dtype=codes.dtype)
            padded[:codes.shape[0], :] = codes
            codes = padded

        
        predict_codes = self.kmeans.predict(codes)
        
        return wav, torch.from_numpy(predict_codes).long()

    def collate(self, batch):
        wavs, codes = zip(*batch)
        
        wavs = list(wavs)
        codes = list(codes)
        
        wav_lengths = [wav.size(-1) for wav in wavs]
        code_lengths = [code.size(0) for code in codes]
        
        # === WAV ===
        wav_frames = min(self.max_samples, *wav_lengths)
        
        collated_wavs = []
        wav_offsets = []
        
        for wav in wavs:
            wav_diff = wav.size(-1) - wav_frames
            wav_offset = random.randint(0, max(0, wav_diff))
            wav = wav[:, wav_offset: wav_offset + wav_frames]
        
            collated_wavs.append(wav)
            wav_offsets.append(wav_offset)
        
        # === ALIGNEMENT CODES ===
        rate = self.label_rate / self.sample_rate
        
        code_offsets = [
            int(round(wav_offset * rate))
            for wav_offset in wav_offsets
        ]
        
        # clamp offsets pour éviter dépassement
        code_offsets = [
            min(offset, length - 1) if length > 0 else 0
            for offset, length in zip(code_offsets, code_lengths)
        ]
        
        code_frames = int(round(wav_frames * rate))
        
        remaining_code_frames = [
            max(0, length - offset)
            for length, offset in zip(code_lengths, code_offsets)
        ]
        code_frames = max(1, min([code_frames] + remaining_code_frames))
        
        # === CODES ===
        collated_codes = []
        
        for code, offset, length in zip(codes, code_offsets, code_lengths):
            if length == 0:
                sliced = torch.zeros(code_frames, dtype=torch.long)
            else:
                end = offset + code_frames
                sliced = code[offset:end]
        
                # padding si trop court
                if sliced.size(0) < code_frames:
                    pad_size = code_frames - sliced.size(0)
                    pad = torch.zeros(pad_size, dtype=code.dtype)
                    sliced = torch.cat([sliced, pad], dim=0)
        
            collated_codes.append(sliced)
        
        # === STACK ===
        wavs = torch.stack(collated_wavs, dim=0)
        codes = torch.stack(collated_codes, dim=0)
        
        return wavs, codes


####################################################################################
# Testing initialize datasets and dataloaders
####################################################################################

if __name__ == "__main__":
    # This import is used for testing    
    from torch.utils.data import DataLoader

    BATCH_SIZE = 32
    
    train_dataset = AcousticUnitsDataset(
        root=Path("/lium/corpus/base/LibriSpeech"),
        root_length=Path("/lium/raid-a/xcoupe/DATA/LibriSpeech"),
        root_discrete=Path("/lium/scratch/xcoupe/DATA/LibriSpeech/encode/"),
        force_train_kmeans=False,
        train=True,
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate,
        batch_size=BATCH_SIZE,
        num_workers=6,
        pin_memory=False,
        shuffle=False,
        drop_last=True,
    )

    validation_dataset = AcousticUnitsDataset(
        root=Path("/lium/corpus/base/LibriSpeech/"),
        root_length=Path("/lium/raid-a/xcoupe/DATA/LibriSpeech"),
        root_discrete=Path("/lium/scratch/xcoupe/DATA/LibriSpeech/encode/"),
        train=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        collate_fn=train_dataset.collate,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=False,
        drop_last=True,
    )

    print("-----|||TRAIN|||-----")
    for wavs, code in train_loader:
        print("=========== Affichage test du premier élément du dataset de train ===========")
        print(wavs)
        print(wavs.shape)
        print(code)
        print(code.shape)
        break

    print("-----|||TEST|||-----")
    for wavs, code in validation_loader:
        print("=========== Affichage test du premier élément du dataset de validation ===========")
        print(wavs)
        print(wavs.shape)
        print(code)
        print(code.shape)
        break

    