# Using Your Own X-ray ZIP Dataset

This guide explains how to use your own X-ray datasets (in ZIP format) with MedRAG's federated learning framework.

## Overview

MedRAG now supports automatic extraction and distribution of X-ray images from ZIP files across federated hospitals. The `load_zip_dataset.py` script handles:

- **Automatic image categorization** based on filenames and folder structure
- **Even distribution** across 4 hospitals (A, B, C, D) 
- **Train/test splitting** (default 80/20)
- **Multiple image formats** (jpg, png, jpeg)
- **Binary and multi-class** classification support

## Step 1: Prepare Your ZIP File

Your ZIP file should contain X-ray images organized in one of these formats:

### Option A: Folder Structure (Recommended)

```
xray_dataset.zip
├── covid/
│   ├── covid_001.jpg
│   ├── covid_002.jpg
│   └── ...
├── normal/
│   ├── normal_001.jpg
│   ├── normal_002.jpg
│   └── ...
└── pneumonia/
    ├── pneumonia_001.jpg
    └── ...
```

### Option B: Flat Structure with Keyword Filenames

```
xray_dataset.zip
├── patient_001_covid.jpg
├── patient_002_normal.jpg
├── patient_003_pneumonia.jpg
└── ...
```

The script recognizes these keywords:
- **covid**: covid, covid19, covid-19, coronavirus
- **normal**: normal, healthy
- **pneumonia**: pneumonia, bacterial, viral
- **tuberculosis**: tb, tuberculosis
- **lung_opacity**: opacity, effusion

## Step 2: Extract and Organize

Run the ZIP dataset loader to automatically extract and organize your data:

```bash
# Basic usage - creates dataset with 4 hospitals
python src/load_zip_dataset.py --zip-file /path/to/xray_dataset.zip --output-dir ./data

# Binary classification only (covid vs normal)
python src/load_zip_dataset.py --zip-file /path/to/xray_dataset.zip --output-dir ./data --binary

# Custom train/test split (70/30)
python src/load_zip_dataset.py --zip-file /path/to/xray_dataset.zip --output-dir ./data --train-split 0.7

# Use different number of hospitals
python src/load_zip_dataset.py --zip-file /path/to/xray_dataset.zip --output-dir ./data --num-hospitals 6
```

## Step 3: Train with Your Dataset

### Using Different Model Architectures

```bash
# ResNet+VGG (Default)
python src/demo_rag_vfl_with_zip.py --datapath ./data --use-rag

# YOLOv5
python src/demo_rag_vfl_with_zip.py --datapath ./data --model-type yolo5 --use-rag

# YOLOv8
python src/demo_rag_vfl_with_zip.py --datapath ./data --model-type yolo8 --use-rag

# Hybrid ResNet + YOLO
python src/demo_rag_vfl_with_zip.py --datapath ./data --model-type resnet_yolo --use-rag

# With Blockchain
python src/demo_rag_vfl_with_zip.py --datapath ./data --model-type yolo5 --use-rag --withblockchain
```

## Step 4: Compare Model Architectures

```bash
python src/demo_rag_vfl_with_zip.py --datapath ./data --compare-models --model-type resnet_vgg
```

## Supported Model Architectures

- **resnet_vgg** (default): ResNet50 + VGG19
- **yolo5**: YOLOv5 feature extraction
- **yolo8**: YOLOv8 feature extraction  
- **resnet_yolo**: Hybrid ResNet50 + YOLO
- **vit**: Vision Transformer

## Requirements

```bash
# For ZIP extraction
pip install Pillow tqdm

# For YOLO models
pip install ultralytics opencv-python

# Or install all dependencies
pip install -r requirements.txt
```

## Troubleshooting

### "No valid images found in ZIP file"
Check that your ZIP contains images in supported formats (jpg, png, jpeg) with recognizable keywords in filenames or folders.

### "ultralytics not available"
Install YOLO dependencies: `pip install ultralytics opencv-python`

### "Could not load dataset"
Verify dataset structure: `ls -la data/SplitCovid19/` should show hospitalA, hospitalB, hospitalC, hospitalD

## Multi-Class Support

MedRAG automatically detects the number of classes from your dataset. The training script inspects the `train/` subdirectory of the first hospital folder (`hospitalA` or `client0`) and derives class names from the folder names (sorted alphabetically).

For example, a dataset with four classes:

```
SplitCovid19/
└── hospitalA/
    └── train/
        ├── covid/
        ├── lung_opacity/
        ├── normal/
        └── pneumonia/
```

will automatically configure `num_classes=4` and use `['covid', 'lung_opacity', 'normal', 'pneumonia']` as class labels — no additional flags required.

For **binary datasets** (e.g. only `covid/` and `normal/`), the script detects `num_classes=2` automatically, preserving backward compatibility.

The detected class names and count are saved in the checkpoint metadata so that `inference.py` can restore them automatically when loading a saved model:

```python
from inference import load_inference_model

# Class names are loaded from the checkpoint automatically
engine = load_inference_model(checkpoint_path='path/to/checkpoint.pt')

# Or pass a dataset directory so inference can detect classes independently
engine = load_inference_model(dataset_dir='./data')
```

## Additional Resources

- **Dataset Preparation**: `DATASET_PREPARATION.md`
- **Demo Guide**: `DEMO_GUIDE.md`
- **RAG Documentation**: `src/RAG_README.md`
- **Quick Start**: `QUICKSTART.md`
