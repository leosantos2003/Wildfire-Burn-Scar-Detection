# src/train.py

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# Importa nossas classes customizadas
from data_loader import SiameseDataset
from models import AttentionUNetSiamese
from loss import CompositeLoss

# --- Configurações ---
DATA_ROOT = 'data/processed/dataset/' # Pré-processamento adicionado
T1_DIR = os.path.join(DATA_ROOT, 't1')
T2_DIR = os.path.join(DATA_ROOT, 't2')
MASK_DIR = os.path.join(DATA_ROOT, 'mask')
MODEL_SAVE_PATH = 'models/best_attention_unet.pth'
NUM_EPOCHS = 25 # Comece com 25 e ajuste se necessário
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

def main():
    # Verifica se a GPU está disponível
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # --- Carregamento dos Dados ---
    full_dataset = SiameseDataset(t1_dir=T1_DIR, t2_dir=T2_DIR, mask_dir=MASK_DIR)
    
    # Divide em treino e validação (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Dados carregados: {len(train_dataset)} para treino, {len(val_dataset)} para validação.")

    # --- Inicialização do Modelo, Perda e Otimizador ---
    model = AttentionUNetSiamese(n_channels=9).to(device)
    criterion = CompositeLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) #
    
    best_val_loss = float('inf')

    # --- Loop de Treinamento ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Treino]")
        for t1, t2, mask in progress_bar:
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(t1, t2)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        # --- Loop de Validação ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validação]")
            for t1, t2, mask in progress_bar_val:
                t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
                outputs = model(t1, t2)
                loss = criterion(outputs, mask)
                val_loss += loss.item()
                progress_bar_val.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Treino Loss: {avg_train_loss:.4f}, Validação Loss: {avg_val_loss:.4f}")

        # Salva o melhor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Modelo salvo em {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})")

if __name__ == '__main__':
    main()