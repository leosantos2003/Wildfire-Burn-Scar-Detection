# src/preprocess_features.py

import os
import numpy as np
import rasterio
from tqdm import tqdm

# Reutiliza nossas funções do DataLoader original
from data_loader import normalize_percentile, get_glcm_features
import spyndex

# --- Configuração de Pastas ---
RAW_DATA_DIR = 'data/dataset'
PROCESSED_DATA_DIR = 'data/processed/dataset'

def preprocess_and_save():
    print("Iniciando pré-processamento de features...")
    
    # Itera sobre t1, t2 e mask
    for folder in ['t1', 't2', 'mask']:
        raw_folder_path = os.path.join(RAW_DATA_DIR, folder)
        processed_folder_path = os.path.join(PROCESSED_DATA_DIR, folder)
        os.makedirs(processed_folder_path, exist_ok=True)
        
        print(f"Processando pasta: {raw_folder_path}")
        
        file_list = [f for f in os.listdir(raw_folder_path) if f.endswith('.tif')]
        
        for filename in tqdm(file_list, desc=f"Pasta {folder}"):
            raw_file_path = os.path.join(raw_folder_path, filename)
            
            with rasterio.open(raw_file_path) as src:
                if folder == 'mask':
                    # Processamento simples para máscaras
                    mask = src.read(1).astype(np.float32)
                    mask = np.nan_to_num(mask, nan=0.0)
                    processed_data = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
                else:
                    # Processamento completo para imagens t1 e t2
                    img_raw = src.read().astype(np.float32)
                    img_raw = np.nan_to_num(img_raw, nan=0.0)

                    params = {"N": img_raw[1], "R": img_raw[0], "S1": img_raw[2], "S2": img_raw[3]}
                    epsilon = 1e-6
                    params_safe = {k: v + epsilon for k, v in params.items()}
                    
                    nbr = spyndex.computeIndex(index="NBR", params=params_safe)
                    nbrswir = spyndex.computeIndex(index="NBRSWIR", params=params_safe)
                    
                    glcm_features = get_glcm_features(img_raw[1])

                    all_features = np.vstack([
                        img_raw,
                        nbr.reshape(1, 128, 128),
                        nbrswir.reshape(1, 128, 128),
                        np.broadcast_to(glcm_features[:, np.newaxis, np.newaxis], (3, 128, 128))
                    ]).astype(np.float32)

                    processed_data = normalize_percentile(all_features)

            # Salva o resultado como um arquivo .npy (eficiente para numpy/pytorch)
            save_path = os.path.join(processed_folder_path, filename.replace('.tif', '.npy'))
            np.save(save_path, processed_data)

    print("Pré-processamento concluído!")

if __name__ == '__main__':
    preprocess_and_save()