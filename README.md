# Wildfire Burn Scar Detection

```
/
|-- data/
|   |-- dataset/
|   `-- avaliacao/
|-- src/
|   |-- data_loader.py
|   |-- models.py
|   |-- loss.py
|   |-- train.py
|   `-- evaluate.py
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

1.  **Train Model:** To begin training from scratch:
    ```bash
    python src/train.py
    ```

2.  **Evaluate Model:** To calculate the metrics (Accuracy, IoU, F1, AUC) in the validation set:
    ```bash
    python src/evaluate.py
    ```
