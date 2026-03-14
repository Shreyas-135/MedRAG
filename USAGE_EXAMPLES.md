# MedRAG Enhancements - Usage Examples

This document provides complete, end-to-end examples for using the new ZIP dataset loader and YOLO model features.

## Quick Start: 3-Step Workflow

### Step 1: Extract Your ZIP Dataset

```bash
# Basic usage with binary classification (covid vs normal)
python src/load_zip_dataset.py \
    --zip-file /path/to/your/xray_dataset.zip \
    --output-dir ./data \
    --binary

# Expected output:
# 📦 Extracting ZIP file: xray_dataset.zip
# ✓ Extraction complete
# 🔍 Scanning for X-ray images...
# 📊 Found images by class:
#    - covid: 500 images
#    - normal: 500 images
# 🏥 Distributing images across 4 hospitals...
# ================================================================================
# 📊 DISTRIBUTION SUMMARY
# ================================================================================
# 🏥 Hospital A: 200 train, 50 test images
# 🏥 Hospital B: 200 train, 50 test images
# 🏥 Hospital C: 200 train, 50 test images
# 🏥 Hospital D: 200 train, 50 test images
# ================================================================================
```

### Step 2: Train with Your Preferred Model

```bash
# Option A: ResNet+VGG (Default - Highest Accuracy)
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data \
    --use-rag \
    --num-epochs 10

# Option B: YOLOv5 (Fastest Inference)
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data \
    --model-type yolo5 \
    --use-rag \
    --num-epochs 10

# Option C: Hybrid ResNet+YOLO (Best of Both)
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data \
    --model-type resnet_yolo \
    --use-rag \
    --num-epochs 15

# Expected output:
# ================================================================================
# RAG-Enhanced Vertical Federated Learning for Medical Imaging
# ================================================================================
# Configuration:
#   - Hospitals: A, B, C, D (4 nodes)
#   - Model Architecture: YOLO5
#   - RAG Enhancement: Enabled
#   - Blockchain: Disabled
# ================================================================================
# ✓ Hospital A: 200 train, 50 test samples
# ✓ Hospital B: 200 train, 50 test samples  
# ✓ Hospital C: 200 train, 50 test samples
# ✓ Hospital D: 200 train, 50 test samples
# ✓ YOLO model initialized (7.2M parameters)
# ✓ RAG knowledge base loaded
```

### Step 3: Compare Model Performance

```bash
# Compare different architectures before committing to one
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data \
    --compare-models \
    --model-type resnet_vgg

# Expected output:
# ================================================================================
# MODEL ARCHITECTURE COMPARISON
# ================================================================================
# 
# RESNET_VGG:
#   Parameters: 32,567,488
#   Trainable:  32,567,488
#   Size:       124.23 MB
#   Output:     torch.Size([1, 64])
#   Inference:  45.23 ms
# 
# YOLO5:
#   Parameters: 7,235,389
#   Trainable:  1,890,432
#   Size:       27.60 MB
#   Output:     torch.Size([1, 64])
#   Inference:  28.15 ms
# ================================================================================
```

## Complete Examples

### Example 1: COVID-19 Detection with Binary Classification

```bash
# Scenario: You have a ZIP file with COVID and normal X-rays
# Goal: Train a fast, accurate model for binary classification

# Step 1: Prepare dataset
python src/load_zip_dataset.py \
    --zip-file covid_dataset.zip \
    --output-dir ./data_covid \
    --binary \
    --train-split 0.8

# Step 2: Train with YOLOv5 (fast + accurate)
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data_covid \
    --model-type yolo5 \
    --use-rag \
    --num-epochs 15 \
    --theta 0.1

# Step 3: Train with blockchain for transparency
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data_covid \
    --model-type yolo5 \
    --use-rag \
    --withblockchain \
    --num-epochs 15

# Results will be saved to:
# - model_registry/versions/  (trained models)
# - ledger/  (audit trail)
```

### Example 2: Multi-Class Pneumonia Detection

```bash
# Scenario: ZIP with covid, normal, pneumonia classes
# Goal: Multi-class classification with high accuracy

# Step 1: Prepare dataset (all classes)
python src/load_zip_dataset.py \
    --zip-file pneumonia_dataset.zip \
    --output-dir ./data_pneumonia \
    --train-split 0.85

# Step 2: Compare models first
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data_pneumonia \
    --compare-models

# Step 3: Train with best model (hybrid for multi-class)
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data_pneumonia \
    --model-type resnet_yolo \
    --use-rag \
    --num-epochs 20 \
    --enable-uncertainty
```

### Example 3: Large Dataset with Custom Hospitals

```bash
# Scenario: Large dataset to distribute across 6 hospitals
# Goal: Scale federated learning to more nodes

# Step 1: Distribute across 6 hospitals
python src/load_zip_dataset.py \
    --zip-file large_xray_dataset.zip \
    --output-dir ./data_large \
    --num-hospitals 6 \
    --train-split 0.85

# Step 2: Train with all features
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data_large \
    --model-type yolo8 \
    --use-rag \
    --withblockchain \
    --num-epochs 25 \
    --enable-uncertainty \
    --enable-explainability
```

### Example 4: Testing Different Dataset Sizes

```bash
# Prepare dataset once
python src/load_zip_dataset.py \
    --zip-file xray_dataset.zip \
    --output-dir ./data

# Test with different dataset fractions
# Small dataset (1.25% - for quick testing)
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data \
    --model-type yolo5 \
    --datasize 0.0125 \
    --num-epochs 3

# Medium dataset (50%)
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data \
    --model-type resnet_yolo \
    --datasize 0.5 \
    --num-epochs 10

# Full dataset (100%)
python src/demo_rag_vfl_with_zip.py \
    --datapath ./data \
    --model-type resnet_yolo \
    --use-rag \
    --withblockchain \
    --datasize 1.0 \
    --num-epochs 20
```

## Command-Line Options Reference

### load_zip_dataset.py Options

```bash
python src/load_zip_dataset.py \
    --zip-file PATH          # Path to ZIP file (required)
    --output-dir PATH        # Output directory (default: ./data)
    --num-hospitals N        # Number of hospitals (default: 4)
    --train-split RATIO      # Train/test split ratio (default: 0.8)
    --binary                 # Use only covid and normal classes
    --seed SEED              # Random seed for reproducibility (default: 42)
```

### demo_rag_vfl_with_zip.py Options

```bash
python src/demo_rag_vfl_with_zip.py \
    --datapath PATH               # Path to dataset (required)
    --model-type TYPE             # Model architecture (choices below)
    --use-rag                     # Enable RAG enhancement
    --withblockchain              # Enable blockchain aggregation
    --num-epochs N                # Number of training epochs (default: 5)
    --datasize FRACTION           # Dataset size fraction (0.0125, 0.25, 0.5, 1.0)
    --theta VALUE                 # DP noise parameter (default: 0.1)
    --compare-models              # Compare architectures before training
    --enable-uncertainty          # Enable uncertainty quantification
    --enable-explainability       # Enable explainability features
```

**Model Types:**
- `resnet_vgg` - ResNet50 + VGG19 (default, highest accuracy)
- `yolo5` - YOLOv5 (fast inference)
- `yolo8` - YOLOv8 (latest YOLO)
- `resnet_yolo` - Hybrid ResNet + YOLO (balanced)
- `vit` - Vision Transformer
- `vit_small` - Small Vision Transformer
- `hybrid` - CNN + ViT hybrid

## Performance Comparison

Based on typical X-ray datasets:

| Model | Parameters | Size | Inference Time | Accuracy | Best For |
|-------|-----------|------|----------------|----------|----------|
| resnet_vgg | 32.6M | 124 MB | 45ms | Highest | Accuracy-critical |
| yolo5 | 7.2M | 28 MB | 28ms | High | Fast inference |
| yolo8 | 11.1M | 42 MB | 32ms | High | Latest features |
| resnet_yolo | 25.8M | 99 MB | 52ms | Highest | Balanced approach |

## Troubleshooting Common Issues

### Issue: "No valid images found in ZIP file"

**Solution:**
```bash
# Check ZIP contents
unzip -l your_dataset.zip

# Ensure filenames/folders contain keywords:
# covid, normal, pneumonia, tuberculosis, etc.

# Or organize ZIP with folders:
# covid/image001.jpg
# normal/image001.jpg
```

### Issue: "ultralytics not available"

**Solution:**
```bash
# Install YOLO dependencies
pip install ultralytics opencv-python

# Verify installation
python -c "from ultralytics import YOLO; print('✓ YOLO available')"
```

### Issue: "Could not load dataset"

**Solution:**
```bash
# Verify structure
ls -la data/SplitCovid19/
# Should show: hospitalA, hospitalB, hospitalC, hospitalD

# Check one hospital
ls -la data/SplitCovid19/hospitalA/
# Should show: train, test

# Check classes
ls -la data/SplitCovid19/hospitalA/train/
# Should show: covid, normal, etc.
```

### Issue: YOLO model downloads slowly on first use

**Solution:**
```bash
# Pre-download YOLO models
python -c "
from ultralytics import YOLO
print('Downloading YOLOv5...')
YOLO('yolov5n.pt')
print('Downloading YOLOv8...')
YOLO('yolov8n.pt')
print('✓ Models cached')
"

# Models are now cached for future use
```

## Integration with Web UI

After training, visualize results in the web dashboard:

```bash
# Launch web interface
streamlit run webapp/app.py

# Then navigate to:
# - http://localhost:8501 (main dashboard)
# - Inference page: Upload X-rays for predictions
# - Training page: View training metrics
# - Registry page: Browse saved models
# - Ledger page: Review audit trail
```

## Best Practices

1. **Dataset Preparation**
   - Use consistent naming: covid, normal, pneumonia
   - Balance classes (similar number of images per class)
   - High quality images (≥224x224 pixels)

2. **Model Selection**
   - Small datasets (<500 images): `resnet_vgg`
   - Fast inference needed: `yolo5`
   - Production deployment: `resnet_yolo`
   - Research/experimentation: Compare all models

3. **Training Parameters**
   - Start with small datasize (0.0125) for quick testing
   - Use RAG (`--use-rag`) for better explainability
   - Enable blockchain for transparency
   - Increase epochs for larger datasets (20-30)

4. **Hospital Distribution**
   - 4 hospitals: Standard federated learning
   - 6-8 hospitals: Large-scale collaboration
   - More hospitals = more privacy, but slower training

## Additional Resources

- **Documentation**: See `docs/USING_YOUR_XRAY_ZIP.md`
- **Dataset Guide**: See `DATASET_PREPARATION.md`
- **Demo Guide**: See `DEMO_GUIDE.md`
- **RAG Details**: See `src/RAG_README.md`

## Support

For issues:
1. Check error messages carefully
2. Verify dataset structure
3. Ensure dependencies are installed
4. Review examples above
5. Check GitHub issues
