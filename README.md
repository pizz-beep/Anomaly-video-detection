# Video Anomaly Detection via LSTM + Adaptive Thresholding

This repository contains the implementation for **"Stabilizing Threshold-Based Video Anomaly Detection via Reinforcement Learning"**, a research project from the CCBD-CDSAML Lab at PES University.

The project treats anomaly detection as a two-stage problem — anomaly *scoring* and anomaly *decision-making* — and evaluates both components independently across two benchmark datasets.

---

## Overview

Standard reconstruction-based anomaly detectors produce noisy frame-level error signals that are highly sensitive to the choice of threshold. This work addresses that instability by replacing static thresholds with adaptive decision controllers.

Two pipelines are implemented:

| Dataset | Feature Extraction | LSTM Type | Decision Mechanism |
|---|---|---|---|
| UCSD Ped2 | ResNet-18 + Optical Flow | Single-layer, hidden=256 | Feedback-driven adaptive threshold |
| Avenue | YOLOv8 + DeepSORT tracks | 3-layer, hidden=256, dropout=0.3 | DQN-based RL confidence modulator |

This notebook covers the **UCSD Ped2 pipeline** end-to-end.

---

## Results

### UCSD Ped2

| Metric | Value |
|---|---|
| AUC (raw) | 0.638 |
| AUC (smoothed) | 0.649 |
| F1-score (RL threshold) | 0.643 |
| Precision | 0.916 |
| Recall | 0.495 |

The adaptive threshold achieves high precision (few false positives) at the cost of recall — short anomalous bursts are sometimes missed.

### Avenue (reported in paper)

| Method | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Adaptive threshold baseline | 0.205 | 0.123 | 0.153 | 0.470 |
| RL (DQN) controller | 0.241 | 0.146 | 0.182 | 0.493 |

---

## Pipeline — UCSD Ped2

### 1. Feature Extraction

Each video frame is processed to produce a fused spatiotemporal representation:

- **Appearance**: ResNet-18 (pretrained on ImageNet, final FC removed) extracts a 512-d embedding from the RGB frame.
- **Motion**: Dense optical flow computed between consecutive frames using Farneback's method. Magnitude and angle components are derived via polar conversion.

These two signals are combined per frame before being passed to the LSTM.

### 2. LSTM Temporal Prediction Model

A single-layer LSTM (hidden size 256) is trained in a self-supervised manner on **normal frames only**. The model learns to predict the next frame's feature vector given the current one. At inference time, the mean squared error between the predicted and actual feature vector serves as the **frame-level anomaly score** — higher error signals deviation from learned normal dynamics.

### 3. Adaptive Threshold Agent

Raw scores are first smoothed using a sliding window average (window=7). An `RLThresholdAgent` then processes each score sequentially:

```
threshold_{t+1} = threshold_t + α * (score_t - threshold_t)
```

The threshold adapts to track the score distribution over time. A frame is flagged as anomalous when the smoothed score exceeds the current threshold. This feedback-driven update is RL-inspired — the "reward" is the signed difference between the score and the threshold, encouraging separation between normal and anomalous frames.

---

## Repository Structure

```
.
├── ResNet_LSTM.ipynb       # Full Ped2 pipeline (feature extraction → LSTM → adaptive threshold)
└── README.md
```

The Avenue pipeline (YOLOv8 + DeepSORT + 3-layer LSTM + DQN controller) is described in the paper but not included in this notebook.

---

## Setup

### Requirements

```bash
pip install torch torchvision opencv-python scikit-learn tqdm matplotlib kaggle
```

### Dataset

The notebook uses the UCSD Ped2 dataset, available on Kaggle:

```bash
kaggle datasets download -d karthiknm1/ucsd-anomaly-detection-dataset
unzip ucsd-anomaly-detection-dataset.zip
```

You will need a `kaggle.json` API token. Place it at `~/.kaggle/kaggle.json`.

Expected directory layout after extraction:

```
UCSD_Anomaly_Dataset.v1p2/
└── UCSDped2/
    ├── Train/
    │   ├── Train001/
    │   └── ...
    └── Test/
        ├── Test001/
        ├── Test001_gt/
        └── ...
```

### Running

Open `ResNet_LSTM.ipynb` in Google Colab (recommended for GPU access) and run cells in order. The notebook will:

1. Install dependencies and configure Kaggle
2. Download and extract the dataset
3. Extract spatiotemporal features from all training clips
4. Train the LSTM prediction model for 20 epochs
5. Score all test frames and load ground truth labels
6. Plot the ROC curve and anomaly score vs ground truth
7. Apply the adaptive threshold agent and report F1, precision, recall, and AUC

---

## Key Design Decisions

**Why separate scoring from decision-making?** The LSTM is never retrained — the adaptive threshold operates purely at inference time. This means the decision controller can be swapped or improved independently of the underlying model.

**Why ResNet-18?** It provides strong generalizable appearance features without fine-tuning. The pretrained weights transfer well to pedestrian surveillance scenes.

**Why Farneback optical flow?** It is dense (covers all pixels), computationally efficient, and captures subtle motion patterns relevant to anomaly detection — such as a bicycle in a pedestrian zone.

**Why a feedback-driven threshold instead of a fixed one?** Fixed thresholds require manual tuning per dataset. The adaptive agent tracks the score distribution in real time, making it more robust to scene-level variation in reconstruction error magnitude.

---

## Citation

If you use this work, please cite:

```
Jayapriya D, Siya Moghe, Dennis Philip, Alisha Prakash, Aditya Hubli,
"Stabilizing Threshold-Based Video Anomaly Detection via Reinforcement Learning,"
PES University, CCBD-CDSAML Lab, 2025. (submitted)
```

---

## Acknowledgements

This work was supported by the Center for Cloud Computing & Big Data (CCBD) and the Centre for Data Science and Machine Learning (CDSAML) at PES University, Bangalore, India.
