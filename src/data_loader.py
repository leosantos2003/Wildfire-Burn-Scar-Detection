# src/data_loader.py

import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

class SiameseDataset(Dataset):
    """
    Classe de Dataset para carregar pares de imagens (T1, T2) e a máscara de referência
    para a detecção de mudanças bitemporais.
    Baseado no código fornecido no baseline do WorCap 2025.
    """
    def __init__(self, t1_dir, t2_dir, mask_dir, transform=None):
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Garante que estamos processando apenas imagens que existem nos três diretórios
        self.ids = sorted([
            f.replace('recorte_', '').replace('.tif', '')
            for f in os.listdir(t1_dir)
            if f.startswith('recorte_') and
               os.path.isfile(os.path.join(t2_dir, f)) and
               os.path.isfile(os.path.join(mask_dir, f))
        ])

    def __len__(self):
        return len(self.ids)

    def read_image(self, path):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
            # Tratamento de valores nulos e normalização simples
            img = np.nan_to_num(img, nan=0.0)
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)
        return torch.tensor(img, dtype=torch.float32)

    def read_mask(self, path):
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.float32)
            mask = np.nan_to_num(mask, nan=0.0)
            # Binariza a máscara para garantir valores 0 ou 1
            mask = np.where(mask > 0, 1.0, 0.0)
        return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        fname = f"recorte_{image_id}.tif"
        
        t1_path = os.path.join(self.t1_dir, fname)
        t2_path = os.path.join(self.t2_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        t1 = self.read_image(t1_path)
        t2 = self.read_image(t2_path)
        mask = self.read_mask(mask_path)

        if self.transform:
            t1 = self.transform(t1)
            t2 = self.transform(t2)
            mask = self.transform(mask)

        return t1, t2, mask