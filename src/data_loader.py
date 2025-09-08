# src/data_loader.py (versão otimizada)

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SiameseDataset(Dataset):
    def __init__(self, t1_dir, t2_dir, mask_dir, augmentation=True):
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.mask_dir = mask_dir
        self.augmentation = augmentation

        self.ids = sorted([
            f.replace('.npy', '').replace('recorte_', '')
            for f in os.listdir(t1_dir)
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        fname = f"recorte_{image_id}.npy"
        
        t1_path = os.path.join(self.t1_dir, fname)
        t2_path = os.path.join(self.t2_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # Carrega os tensores e GARANTE que eles sejam do tipo float32
        t1 = torch.from_numpy(np.load(t1_path)).float() # <-- ADICIONE .float() AQUI
        t2 = torch.from_numpy(np.load(t2_path)).float() # <-- ADICIONE .float() AQUI
        mask = torch.from_numpy(np.load(mask_path)).float().unsqueeze(0) # <-- ADICIONE .float() AQUI
        
        if self.augmentation:
            if torch.rand(1) < 0.5:
                t1, t2, mask = transforms.functional.hflip(t1), transforms.functional.hflip(t2), transforms.functional.hflip(mask)
            if torch.rand(1) < 0.5:
                t1, t2, mask = transforms.functional.vflip(t1), transforms.functional.vflip(t2), transforms.functional.vflip(mask)

        return t1, t2, mask