# RT-DETR training on Windows GPU

This folder contains a Windows-friendly script to train RT-DETR on the ACDC dataset.

## Requirements
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support
- NVIDIA driver installed
- Internet access for downloading the dataset and model weights

## Install dependencies
Open PowerShell in this folder and run:

```powershell
py -m pip install --upgrade pip
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
py -m pip install ultralytics kagglehub
```

If `torch` is already installed and working, you can skip the PyTorch install step.

## Run training

### Train rain and fog together
```powershell
py train_rtdetr_acdc.py --weather both --epochs 10 --imgsz 640 --batch-size 8 --device 0
```

### Train only rain
```powershell
py train_rtdetr_acdc.py --weather rain --epochs 10 --imgsz 640 --batch-size 8 --device 0
```

### Train only fog
```powershell
py train_rtdetr_acdc.py --weather fog --epochs 10 --imgsz 640 --batch-size 8 --device 0
```

## Notes
- The script downloads the dataset from Kaggle using KaggleHub.
- You need Kaggle API credentials configured for the download to work.
- The output will be stored in the `acdc_output` folder by default.
