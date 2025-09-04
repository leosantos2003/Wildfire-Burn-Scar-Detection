# src/data_loader.py

import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import spyndex
import torch.nn.functional as F
from skimage.feature import graycomatrix, graycoprops
from torchvision import transforms

# --- Normalização Avançada ---
def normalize_percentile(image, p_min=1, p_max=99):
    """Normaliza a imagem usando clip de percentil para remover outliers."""
    #
    normalized_bands = []
    for band in image:
        # Remove NaNs para cálculo de percentil
        band_no_nan = band[~np.isnan(band)]
        if band_no_nan.size > 0:
            lower_percentile, upper_percentile = np.percentile(band_no_nan, [p_min, p_max])
            # Clip e normalização
            clipped_band = np.clip(band, lower_percentile, upper_percentile)
            mean = clipped_band.mean()
            std = clipped_band.std()
            if std > 0:
                normalized_band = (clipped_band - mean) / std
            else:
                normalized_band = clipped_band - mean
            normalized_bands.append(normalized_band)
        else:
            normalized_bands.append(band) # Deixa como está se a banda estiver vazia
    return np.stack(normalized_bands)

# --- Features de Textura (GLCM) ---
def get_glcm_features(image_band):
    """Calcula features GLCM para uma única banda da imagem."""
    #
    # Converte para 8-bit, necessário para o GLCM
    band_uint8 = (image_band * 255).astype(np.uint8)
    
    glcm = graycomatrix(band_uint8, distances=[1], angles=[0], symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return np.array([contrast, homogeneity, correlation], dtype=np.float32)

class SiameseDataset(Dataset):
    def __init__(self, t1_dir, t2_dir, mask_dir, augmentation=True):
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.mask_dir = mask_dir
        self.augmentation = augmentation

        self.ids = sorted([
            f.replace('recorte_', '').replace('.tif', '')
            for f in os.listdir(t1_dir)
            if f.startswith('recorte_') and
               os.path.isfile(os.path.join(t2_dir, f)) and
               os.path.isfile(os.path.join(mask_dir, f))
        ])

    def __len__(self):
        return len(self.ids)

    def read_and_preprocess(self, path):
        with rasterio.open(path) as src:
            # Bandas: 0-Red, 1-NIR, 2-SWIR1, 3-SWIR2
            img_raw = src.read().astype(np.float32)
            img_raw = np.nan_to_num(img_raw, nan=0.0)

            # --- Engenharia de Features Espectrais ---
            #
            # Usando spyndex para calcular índices
            params = {
                "N": img_raw[1], "R": img_raw[0],
                "S1": img_raw[2], "S2": img_raw[3]
            }
            # Adiciona um epsilon para evitar divisão por zero
            epsilon = 1e-6
            params_safe = {k: v + epsilon for k, v in params.items()}
            
            nbr = spyndex.computeIndex(index="NBR", params=params_safe)
            nbrswir = spyndex.computeIndex(index="NBRSWIR", params=params_safe)
            
            # --- Engenharia de Features de Textura ---
            #
            # Calculado na banda NIR (índice 1)
            glcm_features = get_glcm_features(img_raw[1])

            # Empilha todas as features: 4 bandas originais + 2 índices + 3 texturas = 9 canais
            all_features = np.vstack([
                img_raw,
                nbr.reshape(1, 128, 128),
                nbrswir.reshape(1, 128, 128),
                np.broadcast_to(glcm_features[:, np.newaxis, np.newaxis], (3, 128, 128))
            ]).astype(np.float32)

            # Normalização avançada
            normalized_features = normalize_percentile(all_features)
            
            return normalized_features

    def read_mask(self, path):
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.float32)
            mask = np.nan_to_num(mask, nan=0.0)
            mask = np.where(mask > 0, 1.0, 0.0)
        return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        fname = f"recorte_{image_id}.tif"
        
        t1_path = os.path.join(self.t1_dir, fname)
        t2_path = os.path.join(self.t2_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        t1 = torch.tensor(self.read_and_preprocess(t1_path), dtype=torch.float32)
        t2 = torch.tensor(self.read_and_preprocess(t2_path), dtype=torch.float32)
        mask = self.read_mask(mask_path)
        
        # --- Aumento de Dados (Augmentation) ---
        #
        if self.augmentation:
            # Aplica a mesma transformação aleatória para t1, t2 e a máscara
            if torch.rand(1) < 0.5:
                t1, t2, mask = transforms.functional.hflip(t1), transforms.functional.hflip(t2), transforms.functional.hflip(mask)
            if torch.rand(1) < 0.5:
                t1, t2, mask = transforms.functional.vflip(t1), transforms.functional.vflip(t2), transforms.functional.vflip(mask)

        return t1, t2, mask