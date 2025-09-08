# Wildfire Burn Scar Detection

## About

This repository contains the solution developed for the WorCap 2025 Hackathon, focused on detecting fire scars in Brazilian biomes using bi-temporal satellite images and Deep Learning techniques.

The mission, as proposed by the event, is to develop a computational solution to map affected areas, contributing to prevention, mitigation, and environmental recovery efforts. Starting from a simple baseline, this project evolved into a robust and optimized pipeline, incorporating feature engineering, an advanced neural network architecture, and a workflow focused on performance and reproducibility.

## Development Phases

* **Phase 1**: The project began with setting up a robust development environment and creating an initial DataLoader to load and visualize the raw data, ensuring a solid and reproducible foundation.

* **Phase 2**: To enrich the information available to the model, a crucial feature engineering step was implemented. The goal was to inject remote sensing "domain knowledge" directly into the data, calculating spectral and textural indices known to be effective in detecting fires.

* **Phase 3**: The core architecture was developed, a Siamese Attention U-Net, and the training loop. This phase focused on building a model capable of learning the complex spatial and temporal relationships of the enriched data and training it efficiently.

* **Phase 4**: After the first training session, a speed bottleneck was identified. Calculating features "live" every epoch was slowing the process down. The solution was to create a preprocessing pipeline where all heavy calculations are performed once and saved to disk. This resulted in a drastic acceleration in training time, from several minutes to seconds per epoch.

## Key-concepts and calculus explained

1. **Model Architecture: Siamese Attention U-Net**

The choice of architecture was deliberate to solve the bi-temporal change detection problem.

* **Siamese Architecture**: Consists of two identical encoders that process T1 (before) and T2 (after) images in parallel. Given that the encoders share the same weights, they learn to map both images to the same "feature space," allowing for meaningful comparison between them.

* **U-Net Backbone**: U-Net is a classic architecture for image segmentation. Its encoder-decoder structure with skip connections allows the model to capture both the overall image context (in the deeper encoder layers) and the fine localization details (preserved by the skip connections), which is essential for generating prediction masks with accurate edges.

* **Attention Mechanism (CBAM)**: To enhance U-Net, we integrated a Convolutional Block Attention Module (CBAM). Attention allows the model to learn to "pay more attention" to the most informative regions and features of the image. It does this in two steps:

    * Channel Attention: Decide which channels (e.g., is the NIR band more important than the Red band?) are more relevant.
    * Spatial Attention: Decide which areas of the image (e.g., the central region with texture changes) are most important. In this model, attention is applied to the absolute difference between the T1 and T2 features, forcing the model to focus precisely on the areas of change.

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
