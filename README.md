# Wildfire Burn Scar Detection

## About

This repository contains the solution developed for the WorCap 2025 Hackathon, focused on detecting fire scars in Brazilian biomes using bi-temporal satellite images and Deep Learning techniques.

The mission, as proposed by the event, is to develop a computational solution to map affected areas, contributing to prevention, mitigation, and environmental recovery efforts. Starting from a simple baseline, this project evolved into a robust and optimized pipeline, incorporating feature engineering, an advanced neural network architecture, and a workflow focused on performance and reproducibility.

graph TD
    A[1. Dados Brutos (.tif)] -- "Imagens T1, T2 e Máscaras" --> B(2. Pré-processamento);
    subgraph 2. Pré-processamento [src/preprocess_features.py]
        B1(Cálculo de Índices Espectrais)
        B2(Cálculo de Features de Textura GLCM)
        B3(Normalização por Percentil)
    end
    B -- "Salva Tensores de 9 Canais (.npy)" --> C[3. Dados Processados];
    C -- "Lidos pelo DataLoader Otimizado" --> D(4. Treinamento do Modelo);
    subgraph 4. Treinamento do Modelo [src/train.py]
        D1[Modelo AttentionUNetSiamese]
        D2[Loss Composta (Dice+Focal)]
        D3[Otimizador AdamW]
    end
    D -- "Salva o Melhor Checkpoint (.pth)" --> E[5. Modelo Salvo];
    E -- "Carregado para Avaliação" --> F(6. Avaliação e Análise);
    subgraph 6. Avaliação e Análise [src/evaluate.py + Notebook]
        F1(Cálculo de Métricas)
        F2(Geração de Imagens de Previsão)
    end

```
/
|-- data/
|   |-- dataset/
|   |-- processed/
|   `-- avaliacao/
|-- src/
|   |-- data_loader.py
|   |-- models.py
|   |-- loss.py
|   |-- train.py
|   |-- evaluate.py
|   `-- preprocess_features.py
|   
|-- notebooks/
|-- .gitignore
`-- requirements.txt
```

## How to install

1. **Clone the repository:**
    ```bash
    git clone github.com/leosantos2003/Wildfire-Burn-Scar-Detection
    cd worcap-2025-hackathon
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows:
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to use

1.  **Preprocess the Features:** This step calculates spectral and texture features to speed up training (this step may take a few minutes).
    ```bash
    python src/preprocess_features.py
    ```

2.  **Train Model:** To begin training from scratch:
    ```bash
    python src/train.py
    ```

3.  **Evaluate Model:** To calculate the metrics (Accuracy, IoU, F1, AUC) in the validation set:
    ```bash
    python src/evaluate.py
    ```
