# Gestion du dataset LibriSpeech

Ce dossier permet de gérer le dataset LibriSpeech afin de le rendre compatible avec l'utilisation du code de chargement du dataset 
Le fichier `utils.py` permet de créer le fichier `length.json` qui contient un ensemble de clé-valeur, avec en clé le chemin d'un enregistrement et en valeur le nombre d'échantillon.

Le fichier `encode_dataset` permet de faire d'encoder le dataset LibriSpeech.

Il existe un script bash par fichier qui peut lancer avec `sbatch nom_fichier.sh`. Il faut juste adapter les informations en fonction des cas (modifier le mail, adapter le temps limite...).