# Foreign Object Detection in Wireless EV Charging using Machine Learning

This repository contains the machine learning pipeline developed for foreign object detection (FOD) in wireless power transfer (WPT) systems for electric vehicles. The work is associated with a journal paper submitted to IEEE.

## Overview

Small conductive objects (e.g., keys, coins, wrenches) left on wireless charging pads can cause overheating and reduce power transfer efficiency through eddy current induction. This project implements a parameter-based FOD system that analyzes the transmitter coil current signal — requiring no additional hardware beyond the existing WPT infrastructure.

The best-performing model, a **Random Forest classifier**, achieved:
- **F1 Score:** 0.9783
- **ROC AUC:** 0.9966
- **Cohen's Kappa:** 0.9562

## Repository Structure

```
├── feature_extraction.ipynb   # Signal processing and multi-domain feature engineering
├── ml_localization.ipynb      # Model training, feature selection, and performance evaluation
└── README.md
```

## Methodology

### 1. Data Collection
- 3,680 transmitter coil current waveforms captured via Teledyne LeCroy HDO8108R oscilloscope
- 1,680 samples with foreign objects (wrench, bolt, coin, aluminum can); 2,000 baseline samples
- Objects placed across a 5×5 grid with 20 orientations/positions per cell

### 2. Feature Extraction (`feature_extraction.ipynb`)
49 features extracted across three domains:
- **Time-domain:** mean, RMS, peak-to-peak, skewness, kurtosis, crest factor, etc.
- **Spectral-domain:** THD, SNR, spectral centroid, entropy, flatness, harmonic ratios (Welch's PSD)
- **Wavelet-domain:** energy and entropy across 5 decomposition levels (Daubechies 4)

### 3. ML Pipeline (`ml_localization.ipynb`)
- **Preprocessing:** 70/30 stratified train-test split, z-score standardization
- **Feature selection:** Pearson correlation filtering (49 → 25 features) followed by model-specific RFE
- **Models evaluated:** Random Forest, XGBoost, MLP, SVM, KNN, Logistic Regression
- **Tuning:** Exhaustive grid search with 5-fold cross-validation, optimizing weighted F1 Score

## Results

| Model | F1 Score | ROC AUC | Kappa | Brier |
|-------|----------|---------|-------|-------|
| Random Forest | **0.9783** | **0.9966** | **0.9562** | 0.0253 |
| XGBoost | 0.9728 | 0.9961 | 0.9453 | 0.0208 |
| MLP | 0.9728 | 0.9960 | 0.9453 | 0.0210 |
| SVM | 0.9193 | 0.9665 | 0.8373 | 0.0621 |
| KNN | 0.8791 | 0.9348 | 0.7560 | 0.0934 |
| Logistic Regression | 0.6854 | 0.7547 | 0.3686 | 0.2018 |

## Requirements

```bash
pip install numpy pandas scipy scikit-learn xgboost pywavelets matplotlib seaborn jupyter
```

## Citation

If you use this work, please cite the associated IEEE paper:

```
Y. Zhumay, A. Zollanvari, and M. Bagheri, "Foreign Object Detection using Machine
Learning in Wireless Electric Vehicle Charging," Nazarbayev University, 2024.
```

## Acknowledgements

This work was supported by the Collaborative Research Project (CRP), Nazarbayev University under Grant 211123CRP1604, and the Faculty Development Competitive Research Grant (FDCRG) under Grant 201223FD8811.
