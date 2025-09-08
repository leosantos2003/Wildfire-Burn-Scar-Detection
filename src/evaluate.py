# src/evaluate.py

import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
import os

# Importa nossas classes customizadas
from data_loader import SiameseDataset
from models import AttentionUNetSiamese

# --- Configurações ---
# Garanta que está usando a mesma pasta de dados (processada ou não) que usou para treinar
DATA_ROOT = 'data/processed/dataset' # Mude para 'data/dataset/' se não pré-processou
T1_DIR = os.path.join(DATA_ROOT, 't1')
T2_DIR = os.path.join(DATA_ROOT, 't2')
MASK_DIR = os.path.join(DATA_ROOT, 'mask')
MODEL_PATH = 'models/best_attention_unet.pth'
BATCH_SIZE = 16

def calculate_iou(preds, labels, smooth=1e-6):
    """Calcula o Intersection over Union (IoU), também conhecido como Jaccard Index."""
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    intersection = (preds * labels).sum()
    total = (preds + labels).sum()
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # --- Carregamento dos Dados ---
    # Precisamos recriar a mesma divisão de validação usada no treino
    # Para isso, usamos uma semente fixa no gerador do PyTorch
    torch.manual_seed(42) # Usar uma semente garante que a divisão seja a mesma
    full_dataset = SiameseDataset(t1_dir=T1_DIR, t2_dir=T2_DIR, mask_dir=MASK_DIR, augmentation=False)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Dados de validação carregados: {len(val_dataset)} amostras.")

    # --- Carregamento do Modelo ---
    model = AttentionUNetSiamese(n_channels=9).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Coloca o modelo em modo de avaliação
    print("Modelo carregado com sucesso.")

    # --- Avaliação ---
    all_labels = []
    all_preds_prob = []
    all_preds_binary = []
    total_iou = 0.0

    progress_bar = tqdm(val_loader, desc="Avaliando o modelo")
    with torch.no_grad():
        for t1, t2, mask in progress_bar:
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
            
            # Previsão de probabilidade
            pred_prob = model(t1, t2)
            # Previsão binária (0 ou 1)
            pred_binary = (pred_prob > 0.5).float()
            
            # Acumula os resultados para cálculo de métricas
            all_labels.append(mask.cpu().numpy().flatten())
            all_preds_prob.append(pred_prob.cpu().numpy().flatten())
            all_preds_binary.append(pred_binary.cpu().numpy().flatten())
            
            # Calcula IoU para o lote atual e acumula
            batch_iou = calculate_iou(pred_binary, mask)
            total_iou += batch_iou
    
    # Concatena os resultados de todos os lotes
    all_labels = np.concatenate(all_labels)
    all_preds_prob = np.concatenate(all_preds_prob)
    all_preds_binary = np.concatenate(all_preds_binary)
    
    # --- Cálculo Final das Métricas ---
    auc_score = roc_auc_score(all_labels, all_preds_prob)
    accuracy = accuracy_score(all_labels, all_preds_binary)
    f1 = f1_score(all_labels, all_preds_binary)
    avg_iou = total_iou / len(val_loader)

    print("\n--- Resultados da Avaliação no Conjunto de Validação ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"IoU (Intersection over Union): {avg_iou:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_score:.4f}")
    print("---------------------------------------------------------")

if __name__ == '__main__':
    main()