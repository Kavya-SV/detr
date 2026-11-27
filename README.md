# Weather-Adaptive RT-DETR with Chokam Dynamic Token Routing

This repository implements **Chokam**—a weather-aware dynamic token scoring, 3-group routing, and 1D gather–scatter refinement mechanism integrated into **RT-DETR-L** (`ultralytics_src/`) for object detection under adverse weather conditions (**rain**, **fog**, **snow**, and **night**) on the **ACDC** dataset.

---

## Repository Structure

```text
detr/
├── ultralytics_src/               # Modified Ultralytics source with Chokam modules
│   └── ultralytics/nn/
│       ├── modules/chokam.py      # ChokamTokenScorer, SCEWeatherDetector, ChokamRefiner
│       ├── modules/head.py        # RTDETRDecoder with 3-group Chokam routing & STE gating
│       ├── modules/__init__.py    # Module exports
│       └── tasks.py               # RTDETRDetectionModel loss (L_score + L_sparsity)
├── scripts/
│   ├── train_rtdetr_acdc.py       # Per-weather ACDC conversion & RT-DETR/Chokam training
│   ├── train_sce.py               # Multi-weather SCE Stage-1 training on merged ACDC
│   ├── test_chokam.py             # Forward/backward sanity check & routing/gradient check
│   ├── evaluate_weather.py        # Weather classification eval on ACDC val split
│   ├── evaluate_kaggle.py         # Cross-dataset weather eval on BDD100K (Kaggle)
│   ├── weather_dataset.py         # WeatherRTDETRDataset (injects weather_id per image)
│   ├── merge_acdc_dataset.py      # Merges rain/fog/snow/night into chokam_dataset/
│   ├── split_acdc.py              # Splits chokam_dataset/ into 70/20/10 train/val/test
│   └── compare_5epochs.py         # Prints Baseline vs Chokam V1 vs Chokam V2 rain logs
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

### Prerequisites
- Python **3.10+** (tested with PyTorch `2.13.0+cu126` on Windows GPU)
- NVIDIA GPU with CUDA support (recommended; CPU fallback is supported)
- Kaggle API credentials configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`) for automatic dataset download

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kavya-SV/detr.git
   cd detr
   ```

2. **Create and activate a virtual environment**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   *(On Linux/macOS: `source .venv/bin/activate`)*

3. **Install PyTorch with CUDA** (match your CUDA version; e.g., CUDA 12.1+)
   ```powershell
   pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install dependencies and the local modified `ultralytics_src` package (editable mode)**
   ```powershell
   pip install -r requirements.txt
   ```
   > **Important:** Make sure `ultralytics` resolves to `./ultralytics_src` (installed via `-e ./ultralytics_src` in `requirements.txt`), not an upstream PyPI wheel, so the custom `ChokamTokenScorer`, `SCEWeatherDetector`, and `ChokamRefiner` modules are active.

5. **Verify installation & Chokam gradient flow**
   ```powershell
   python scripts/test_chokam.py
   ```

---

## Dataset Download & Preparation

All scripts expect to be run from the repository root (`detr/`).

### 1. Automatic ACDC Download & Per-Weather Conversion
`scripts/train_rtdetr_acdc.py` automatically downloads `khalilusmanuk/acdc-clean-dataset-for-training` via `kagglehub`, maps the 8 ACDC/Cityscapes traffic categories (`person`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`), converts COCO JSON annotations to YOLO `.txt` labels, and generates `acdc_output/<weather>/dataset.yaml`.

If you already have the ACDC dataset locally (containing an `acdc/` subdirectory with `images/` and `labels/`), pass `--dataset-root <path-to-parent-of-acdc>`.

### 2. Multi-Weather Merged Dataset (`chokam_dataset`) for SCE Training
After generating the per-weather folders (`acdc_output/rain`, `acdc_output/fog`, `acdc_output/snow`, `acdc_output/night`), merge them and create a 70% / 20% / 10% train/val/test split:

```powershell
# Merge rain (0), fog (1), snow (2), night (3) and generate weather_labels.csv
python scripts/merge_acdc_dataset.py

# Shuffle (seed=42) and split into 70% train / 20% val / 10% test
python scripts/split_acdc.py
```

This creates `acdc_output/chokam_dataset/` with `images/{train,val,test}`, `labels/{train,val,test}`, and `weather_labels.csv`.

---

## Model Architecture

Chokam extends `RTDETRDecoder` ([head.py](ultralytics_src/ultralytics/nn/modules/head.py)) and `RTDETRDetectionModel` ([tasks.py](ultralytics_src/ultralytics/nn/tasks.py)) via [chokam.py](ultralytics_src/ultralytics/nn/modules/chokam.py):

```text
Backbone + Neck Features [P3, P4, P5] (C = 256)
   │
   ├──► SCEWeatherDetector ──► weather_logits (4 classes: rain/fog/snow/night)
   │                       └──► tau_net ──► dynamic per-image (tau_low, tau_high)
   │
   └──► ChokamTokenScorer (MLP: 256 → 128 → 64 → 1, sigmoid(x + 2.0))
           │
           ├── score >= tau_high        ──► High Group: kept directly
           ├── tau_low <= score < high  ──► Medium Group (1D gather → ChokamRefiner → re-score → STE gate)
           └── score < tau_low          ──► Low Group: soft-scaled by clamp(score, min=0.15, max=1.0)
                                               │
                                               ▼
                                   Scattered [P3, P4, P5] → RT-DETR Decoder
```

### Key Components
1. **`SCEWeatherDetector`** (`chokam.py`):
   - Applies `AdaptiveAvgPool2d(1)` across multi-scale encoder inputs (`P3, P4, P5`, `num_scales=3`, `in_channels=256`), concatenates into a `768`-dim vector, and predicts 4 weather logits via `768 → 128 → 4` with Dropout (`0.1`).
   - `tau_net` (`4 → 16 → 2`) predicts per-image pruning/routing thresholds `(tau_low, tau_high)` with bias offsets `[-1.38, 0.20]` (initialized near `0.20` and `0.55`, ordered via `min`/`max`).
2. **`ChokamTokenScorer`** (`chokam.py`):
   - Scores every spatial token (`(B, 1, H, W)` or 1D `(N, 1)`) using a 3-layer MLP (`256 → 128 → 64 → 1`) with ReLU and shifted sigmoid `sigmoid(score + 2.0)`.
3. **1D Gather–Scatter & `ChokamRefiner`** (`chokam.py`, `head.py`):
   - Only **medium-confidence** tokens (`tau_low <= score < tau_high`) are gathered into a condensed 1D tensor `(N_med, 256)` and passed through `ChokamRefiner` (residual 2-layer MLP with `LayerNorm`), avoiding unnecessary computation on high-confidence and background tokens.
   - Refined tokens are re-scored; recovered tokens (`score >= tau_high`) receive weight `1.0`, while remaining tokens use Straight-Through Estimator (STE) gating floored at `0.15`.
   - Low-score background tokens are softly attenuated by `clamp(score, min=0.15, max=1.0)`.
4. **Training Objective** (`tasks.py` & `train_sce.py`):
   - `RTDETRDetectionModel.loss` adds `lambda_1 * L_score + lambda_3 * L_sparsity` (`lambda_1 = 0.4`, `lambda_3 = 0.01`), where `L_score` is BCE supervision against binary ground-truth bounding-box masks on each feature grid and `L_sparsity` is the mean predicted token score.
   - Set environment variable `USE_CHOKAM=0` to disable Chokam routing and run the unmodified RT-DETR baseline.

---

## Training & Evaluation Commands

Run all commands from the repository root (`detr/`).

### 1. Train Chokam V2 (Per-Weather)
```powershell
# Train on a single weather condition (rain | fog | snow | night)
python scripts/train_rtdetr_acdc.py --weather rain --epochs 50 --imgsz 640 --batch-size 8 --device 0 --project runs/detect/RTDETR_Chokam_V2
python scripts/train_rtdetr_acdc.py --weather fog --epochs 50 --imgsz 640 --batch-size 8 --device 0 --project runs/detect/RTDETR_Chokam_V2
python scripts/train_rtdetr_acdc.py --weather snow --epochs 50 --imgsz 640 --batch-size 8 --device 0 --project runs/detect/RTDETR_Chokam_V2
python scripts/train_rtdetr_acdc.py --weather night --epochs 50 --imgsz 640 --batch-size 8 --device 0 --project runs/detect/RTDETR_Chokam_V2

# Or train all 4 weather domains sequentially
python scripts/train_rtdetr_acdc.py --weather both --epochs 50 --imgsz 640 --batch-size 8 --device 0 --project runs/detect/RTDETR_Chokam_V2
```

### 2. Train Pure RT-DETR Baseline (Chokam Disabled)
```powershell
$env:USE_CHOKAM="0"
python scripts/train_rtdetr_acdc.py --weather rain --epochs 50 --imgsz 640 --batch-size 8 --device 0 --project runs/detect/RTDETR_Baseline_Pure
Remove-Item Env:USE_CHOKAM
```
*(On Linux/macOS: `USE_CHOKAM=0 python scripts/train_rtdetr_acdc.py ...`)*

### 3. Train Multi-Weather SCE Weather Classifier
Trains `WeatherAwareRTDETRModel` on the merged `acdc_output/chokam_dataset/` with true Cross-Entropy backpropagation into `SCEWeatherDetector`:
```powershell
python scripts/train_sce.py --epochs 10 --imgsz 640 --batch-size 8 --device 0 --project runs/detect/RTDETR_SCE_Weather --name train_sce_v3
```

### 4. Evaluation & Comparison
```powershell
# Evaluate SCE weather classification on ACDC validation set (Accuracy, Precision, Recall, F1, Confusion Matrix)
python scripts/evaluate_weather.py

# Cross-dataset weather classification evaluation on Kaggle BDD100K (marquis03/bdd100k-weather-classification)
python scripts/evaluate_kaggle.py

# Compare epoch-by-epoch rain metrics across Baseline Pure, Chokam V1, and Chokam V2
python scripts/compare_5epochs.py
```
> Note: `evaluate_weather.py` and `evaluate_kaggle.py` load `runs/detect/RTDETR_SCE_Weather/train_sce_v3/weights/best.pt` by default.

---

## Results: Baseline vs. Chokam V2 (ACDC)

All metrics below are taken directly from the recorded `results.csv` training logs (`rtdetr-l.pt`, `imgsz=640`, `batch=8`).

### Per-Weather Comparison (`RTDETR_Baseline` vs. `RTDETR_Chokam_V2`)

| Weather | Model / Run | Epochs | Best mAP@50 (Epoch) | Best mAP@50-95 (Epoch) | Final mAP@50 | Final mAP@50-95 | Final Precision | Final Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fog** | Baseline (`RTDETR_Baseline/fog`) | 10 | 0.4442 (e10) | 0.3007 (e9) | 0.4442 | 0.3000 | 0.4749 | 0.4719 |
| **Fog** | Chokam V2 (`RTDETR_Chokam_V2/fog` @ ep 10) | 10 | 0.5228 (e10) | 0.3381 (e1) | 0.5228 | 0.3297 | 0.5064 | 0.4773 |
| **Fog** | **Chokam V2 (`RTDETR_Chokam_V2/fog`)** | **50** | **0.6161 (e21)** | **0.4110 (e49)** | **0.6007** | **0.4048** | **0.7685** | **0.5306** |
| **Night** | Baseline (`RTDETR_Baseline/night`) | 10 | 0.3407 (e9) | 0.1795 (e9) | 0.3294 | 0.1779 | 0.5953 | 0.3075 |
| **Night** | Chokam V2 (`RTDETR_Chokam_V2/night` @ ep 10) | 10 | 0.3559 (e10) | 0.1817 (e4) | 0.3559 | 0.1655 | 0.7063 | 0.2888 |
| **Night** | **Chokam V2 (`RTDETR_Chokam_V2/night`)** | **50** | **0.4302 (e19)** | **0.2314 (e19)** | **0.3952** | **0.2107** | **0.6563** | **0.3658** |
| **Snow** | Baseline (`RTDETR_Baseline/snow`) | 10 | 0.4230 (e10) | 0.2694 (e10) | 0.4230 | 0.2694 | 0.5615 | 0.4616 |
| **Snow** | Chokam V2 (`RTDETR_Chokam_V2/snow` @ ep 10) | 10 | 0.4601 (e9) | 0.2695 (e9) | 0.4206 | 0.2431 | 0.6855 | 0.3561 |
| **Snow** | **Chokam V2 (`RTDETR_Chokam_V2/snow`)** | **50** | **0.5022 (e16)** | **0.3133 (e40)** | **0.4614** | **0.3015** | 0.4285 | **0.5269** |
| **Rain** | Baseline Pure (`USE_CHOKAM=0`, 50 ep) | 50 | **0.4392 (e43)** | **0.2755 (e20)** | **0.4192** | **0.2625** | **0.8102** | 0.3656 |
| **Rain** | Chokam V1 (`RTDETR_Chokam/rain`) | 50 | 0.3959 (e40) | 0.2425 (e48) | 0.3874 | 0.2378 | 0.6745 | 0.3602 |
| **Rain** | Chokam V2 (`RTDETR_Chokam_V2/rain`) | 50 | 0.4376 (e39) | 0.2716 (e45) | 0.4023 | 0.2528 | 0.6722 | **0.3716** |

### Multi-Weather Merged SCE Run (`RTDETR_SCE_Weather/train_sce_v3`, 10 Epochs)
- **mAP@50:** `0.5033` | **mAP@50-95:** `0.2984` | **Precision:** `0.6379` | **Recall:** `0.4781`

---

## License & Ultralytics Attribution

- This project builds on **[Ultralytics](https://github.com/ultralytics/ultralytics)** (`ultralytics_src/`), including its RT-DETR implementation (`RTDETRDecoder`, `RTDETRDetectionModel`, `RTDETRTrainer`, `RTDETRValidator`, and `RTDETRDataset`).
- Ultralytics is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** (see [ultralytics_src/LICENSE](ultralytics_src/LICENSE) and [ultralytics_src/CITATION.cff](ultralytics_src/CITATION.cff)).
- Because this repository redistributes and modifies `ultralytics_src/`, all modifications and derivative code in this repository are distributed under the terms of **AGPL-3.0**.
