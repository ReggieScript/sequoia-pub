import os
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import h5py


class SuperTileRNADataset(Dataset):
    def __init__(self, csv_path: str, features_path, feature_use, quick=None):
        self.csv_path = csv_path
        self.quick = quick
        self.features_path = features_path
        self.feature_use = feature_use
        if type(csv_path) == str:
            self.data = pd.read_csv(csv_path)
        else:
            self.data = csv_path

        # find the number of genes
        row = self.data.iloc[0]
        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        self.num_genes = len(rna_data)

        # find the feature dimension, assume all images in the reference file have the same dimension
        wsi = row['wsi_file_name']
        proj = row.get('tcga_project', '')

        # build candidate paths (project subfolder may or may not be used)
        candidates = []
        if proj:
            candidates.append(os.path.join(self.features_path, proj, wsi, wsi + '.h5'))
        candidates.append(os.path.join(self.features_path, wsi, wsi + '.h5'))

        path = None
        for p in candidates:
            if os.path.exists(p):
                path = p
                break
        if path is None:
            raise FileNotFoundError(
                f"Cannot locate feature file for slide {wsi}. Tried: {candidates}")

        print(f"SuperTileRNADataset init using feature file: {path}")
        try:
            f = h5py.File(path, 'r')
        except Exception as e:
            raise FileNotFoundError(
                f"Unable to open feature file {path}. "
                f"You may need to run the feature-extraction stage or filter your reference data. Original exception: {e}")
        # TODO: read actual dimension rather than hardcode 1
        self.feature_dim = 1
        f.close()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        wsi = row['wsi_file_name']
        proj = row.get('tcga_project', '')
        # build candidate paths exactly as in __init__
        candidates = []
        if proj:
            candidates.append(os.path.join(self.features_path, proj, wsi, wsi + '.h5'))
        candidates.append(os.path.join(self.features_path, wsi, wsi + '.h5'))

        path = None
        for p in candidates:
            if os.path.exists(p):
                path = p
                break
        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        rna_data = torch.tensor(rna_data, dtype=torch.float32)
        try:
            if path is None:
                raise FileNotFoundError(f"No feature file found among {candidates}")
            f = h5py.File(path, 'r')
            features = f[self.feature_use][:]
            f.close()
            features = torch.tensor(features, dtype=torch.float32)
        except Exception as e:
            print(e)
            print(path)
            features = None

        return features, rna_data, row['wsi_file_name'], row['tcga_project']