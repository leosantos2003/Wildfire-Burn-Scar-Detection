# src/preprocess_features.py

import os
import numpy as np
import rasterio
from tqdm import tqdm
import spyndex
from skimage.feature import graycomatrix, graycoprops

# --- FUNÇÕES DE AJUDA MOVIDAS PARA CÁ ---
def normalize_percentile(image, p_min=1, p_max=99):
    """Normaliza a imagem usando clip de percentil para remover outliers."""
    #
    normalized_bands = []
    for band in image:
        band_no_nan = band[~np.isnan(band)]
        if band_no_nan.size > 0:
            lower_percentile, upper_percentile = np.percentile(band_no_nan, [p_min, p_max])
            clipped_band = np.clip(band, lower_percentile, upper_percentile)
            mean = clipped_band.mean()
            std = clipped_band.std()
            if std > 0:
                normalized_band = (clipped_band - mean) / std
            else:
                normalized_band = clipped_band - mean
            normalized_bands.append(normalized_band)
        else:
            normalized_bands.append(band)
    return np.stack(normalized_bands)

def get_glcm_features(image_band):
    """Calcula features GLCM para uma única banda da imagem."""
    #
    band_uint8 = (image_band * 255).astype(np.uint8)
    glcm = graycomatrix(band_uint8, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return np.array([contrast, homogeneity, correlation], dtype=np.float32)
# -----------------------------------------


# --- Configuração de Pastas ---
RAW_DATA_DIR = 'data/dataset'
PROCESSED_DATA_DIR = 'data/processed/dataset'

def preprocess_and_save():
    print("Iniciando pré-processamento de features...")
    
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
                    mask = src.read(1).astype(np.float32)
                    mask = np.nan_to_num(mask, nan=0.0)
                    processed_data = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
                else:
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

            save_path = os.path.join(processed_folder_path, filename.replace('.tif', '.npy'))
            np.save(save_path, processed_data)

    print("Pré-processamento concluído!")

if __name__ == '__main__':
    preprocess_and_save()