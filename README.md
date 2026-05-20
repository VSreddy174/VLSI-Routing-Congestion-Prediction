# CNN Architectures for VLSI Routing Congestion Prediction

An extensive ablation study exploring the limits of grid-based Deep Learning for predicting routing hotspots in physical chip layouts.

## Project Overview

Predicting routing congestion during the early stages of VLSI physical design is critical for avoiding costly design rule check (DRC) violations and unroutable chips. This project investigates the mapping of pre-placement features (macro placement, cell density, and RUDY) to final Global Routing (GR) overflow.

Through a series of 6 systematic experiments, this repository evaluates the trade-offs between structural spatial accuracy (measured by SSIM) and actual hotspot detection sensitivity (measured by F1-Score and ROC-AUC), ultimately transitioning from pure regression methodologies to imbalanced classification strategies.

## Experiment Summary & Results

The study progresses from a standard U-Net baseline through decoupled architectures, attention mechanisms, and finally to a multi-scale Inception U-Net formulated as a binary anomaly detection task. 

### Architecture and Strategy

| Experiment | Architecture Strategy | Focus |
| :--- | :--- | :--- |
| Exp 1 | Baseline U-Net (MSE) | General baseline regression. |
| Exp 2 | U-Net + Weighted L1 | High structural accuracy, poor hotspot detection. |
| Exp 3 | Decoupled U-Net (Huber) | Separation of routing channels. |
| Exp 4 | Attention U-Net (CBAM) | Focus on high-traffic spatial corridors. |
| Exp 5 | Decoupled (Binary Pivot) | Shift to weighted classification for sparsity. |
| Exp 6 | Inception U-Net (BCE) | Multi-scale feature extraction (Best hotspot detection). |

### Horizontal (H) Routing Metrics

| Experiment | SSIM | ROC-AUC | F1-Score |
| :--- | :---: | :---: | :---: |
| Exp 1 | 0.758 | 0.892 | 0.320 |
| Exp 2 | 0.969 | 0.757 | 0.283 |
| Exp 3 | 0.792 | 0.895 | 0.382 |
| Exp 4 | 0.684 | 0.931 | 0.399 |
| Exp 5 | 0.410 | 0.966 | 0.464 |
| Exp 6 | 0.443 | 0.975 | 0.502 |

### Vertical (V) Routing Metrics

| Experiment | SSIM | ROC-AUC | F1-Score |
| :--- | :---: | :---: | :---: |
| Exp 1 | 0.788 | 0.914 | 0.314 |
| Exp 2 | 0.921 | 0.881 | 0.231 |
| Exp 3 | 0.823 | 0.916 | 0.389 |
| Exp 4 | 0.728 | 0.925 | 0.415 |
| Exp 5 | 0.477 | 0.932 | 0.388 |
| Exp 6 | 0.378 | 0.940 | 0.435 |

## Data Assets

Data preparation is a massive component of this workflow, involving the zero-padding of varying chip dimensions into consistent 320x320 grid tensors.

* **Original Raw Data:** The original CircuitNet N28 dataset can be found via their official release here: [CircuitNet Google Drive](https://drive.google.com/drive/u/1/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB).
* **Preprocessed Dataset:** The ready-to-train batched shards are hosted at [Kaggle: CircuitNet Preprocessed Features](https://www.kaggle.com/datasets/chupparetti/circuitnet-preprocessed-features/). The dataset contains 103 .npz shards representing 10,242 designs.

## Documentation Guide

The project's methodology, architecture details, and data engineering processes are thoroughly documented in the `docs/` directory. 

* **[`docs/dataset_and_features.md`](docs/dataset_and_features.md)**: Explains the physical meaning of the input features (macro regions, cell density, RUDY variants) and the output targets. It details the masking logic used for grid normalization.
* **[`docs/architectures.md`](docs/architectures.md)**: Details the structural evolution of the models throughout the ablation study, from the Baseline U-Net to the final multi-scale Inception design.
* **[`docs/training_and_loss.md`](docs/training_and_loss.md)**: Covers the mathematical strategies employed to handle severe congestion sparsity, documenting the progression from continuous regression losses to imbalanced binary classification.
* **[`docs/evaluation_and_results.md`](docs/evaluation_and_results.md)**: Contains the comprehensive global evaluation metrics alongside visual inference analysis, including Best, Worst, and Random sample predictions.
