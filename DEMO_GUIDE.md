# MedRAG Capstone Demo Guide

Complete demonstration guide for the MedRAG system - a blockchain-enabled vertical federated learning framework with verifiable RAG for privacy-preserving medical imaging.

## 🎯 Overview

This demo showcases:
- ✅ **Federated Learning**: Privacy-preserving cross-hospital collaboration
- ✅ **Blockchain Integration**: Transparent and immutable model aggregation
- ✅ **RAG Enhancement**: Medical knowledge retrieval for improved predictions
- ✅ **Model Registry**: Version control and checkpoint management
- ✅ **Audit Ledger**: Comprehensive training and access logging
- ✅ **Web Interface**: Professional clinician-facing UI

## 🚀 Quick Start (5 Minutes)

### One-Command Demo

```bash
./run_demo.sh
```

This single command will:
1. Create synthetic demo dataset (50 samples, ~30 seconds)
2. Train RAG-enhanced VFL model (3 epochs, ~2 minutes)
3. Launch web interface (http://localhost:8501)

### Custom Configuration

```bash
# More training epochs
./run_demo.sh --num-epochs 5

# With blockchain integration
./run_demo.sh --with-blockchain

# Skip training if already done
./run_demo.sh --skip-training

# Larger dataset
./run_demo.sh --datasize 0.25 --num-epochs 5
```

## 📋 Demo Walkthrough for Reviewers

### Part 1: System Architecture (2 minutes)

**Navigate to**: Home page (http://localhost:8501)

**What to show**:
- System overview and key components
- Architecture diagram
- Integration of VFL + Blockchain + RAG

**Key points**:
- Privacy-preserving: No raw data sharing between hospitals
- Blockchain: Transparent model aggregation
- RAG: Enhanced predictions with medical knowledge

---

### Part 2: Training Dashboard (3 minutes)

**Navigate to**: 📊 Training Dashboard

**What to show**:
1. **Overview Metrics**
   - Total training rounds completed
   - Model versions created
   - Best accuracy achieved
   - Ledger integrity status

2. **Training Curves**
   - Accuracy improvement over epochs
   - Loss reduction trends
   - Per-client performance (if available)

3. **RAG Statistics**
   - Number of knowledge retrievals
   - Enhancement impact

4. **Privacy Budget**
   - Differential privacy epsilon tracking
   - Cumulative privacy cost

**Key points**:
- Real-time monitoring capability
- Transparent performance metrics
- Privacy budget tracking for compliance

---

### Part 3: Model Registry (3 minutes)

**Navigate to**: 📦 Model Registry

**What to show**:
1. **Registry Overview**
   - All trained model versions
   - Storage utilization
   - Best performing model

2. **Model Details**
   - Select a model version
   - View configuration (RAG, blockchain, privacy settings)
   - Performance metrics (accuracy, loss)
   - Model hash for verification

3. **Model Comparison**
   - Compare 2-3 model versions
   - Side-by-side metric comparison
   - Identify best model

4. **Download Capability**
   - Download model checkpoints
   - Export registry as JSON/CSV

**Key points**:
- Complete version history
- Reproducibility through config tracking
- Model provenance via hashing

---

### Part 4: Inference Demo (5 minutes)

**Navigate to**: 🔬 Inference

**What to show**:
1. **Upload X-ray Image**
   - Use sample from `demo_data/SplitCovid19/client0/test/`
   - Show image preview

2. **Select Model**
   - Choose from available versions
   - Show model info (accuracy, config)

3. **Run Analysis**
   - Click "Analyze Image"
   - Show real-time prediction

4. **Review Results**
   - **Prediction**: COVID-19 vs Normal
   - **Confidence**: Probability score with progress bar
   - **Class Probabilities**: Breakdown for each class
   - **RAG Explanations**: Top 3 retrieved medical findings
   - **Medical Guidelines**: Relevant clinical knowledge
   - **Technical Details**: Inference time, model version

**Key points**:
- Real-time inference
- Explainable AI through RAG
- Clinical decision support
- User-friendly interface

---

### Part 5: Audit Ledger (4 minutes)

**Navigate to**: 📋 Audit Ledger

**What to show**:
1. **Overview**
   - Total entries in ledger
   - Integrity verification status

2. **Training Logs Tab**
   - All training rounds logged
   - Per-round metrics
   - Model hashes
   - Blockchain transaction IDs (if enabled)
   - Privacy budget per round

3. **Access Logs Tab**
   - All system access events
   - Filter by user, action, status
   - Inference predictions logged
   - Training runs logged

4. **Verification Tab**
   - Verify ledger integrity
   - Check hash chain
   - Blockchain verification links

5. **Export**
   - Download logs as CSV/JSON
   - Full audit trail for compliance

**Key points**:
- Immutable audit trail
- Complete transparency
- Compliance ready
- Tamper detection through hash chaining

---

## 🎓 Technical Deep Dive

### Model Registry System

**File**: `src/model_registry.py`

**Features**:
- Version tracking with semantic naming
- SHA-256 model hashing
- Metrics and configuration storage
- Checkpoint management
- Best model selection

**Usage**:
```python
from model_registry import ModelRegistry

registry = ModelRegistry()
version_id = registry.save_model(
    model=server_model,
    round_num=5,
    metrics={'accuracy': 0.85, 'loss': 0.32},
    config={'use_rag': True, 'theta': 0.1}
)

best_model = registry.get_best_model('accuracy')
```

---

### Ledger System

**File**: `src/ledger.py`

**Features**:
- Append-only logs (JSONL format)
- Hash chaining for integrity
- Training round logging
- Access event logging
- Integrity verification

**Usage**:
```python
from ledger import Ledger

ledger = Ledger()

# Log training
ledger.log_training_round(
    round_num=1,
    node_metrics={'client0': {'loss': 0.5}},
    model_hash="abc123...",
    blockchain_tx="0x123...",
    privacy_budget=0.1
)

# Log access
ledger.log_access(
    user_id="doctor_1",
    action="predict",
    resource="xray_001.jpg",
    status="success"
)

# Verify
is_valid = ledger.verify_integrity('training')
```

---

### Inference Engine

**File**: `src/inference.py`

**Features**:
- Single image prediction
- RAG-based explanations
- Confidence scoring
- Medical guideline retrieval

**Usage**:
```python
from inference import load_inference_model

inference = load_inference_model(use_rag=True)
result = inference.predict('path/to/xray.jpg')

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Explanations: {result['explanations']}")
```

---

## 🧪 Testing

### Manual Test Flow

1. **Data Preparation**
   ```bash
   python prepare_dataset.py --output-dir ./demo_data --create-sample --sample-size 50
   ```

2. **Training**
   ```bash
   cd src
   python demo_rag_vfl.py --datapath ../demo_data --use-rag --num-epochs 3
   ```

3. **Verify Registry**
   ```bash
   python -c "from model_registry import ModelRegistry; r = ModelRegistry(); print(r.get_summary())"
   ```

4. **Verify Ledger**
   ```bash
   python -c "from ledger import Ledger; l = Ledger(); print(l.get_summary())"
   ```

5. **Launch Web UI**
   ```bash
   cd webapp
   streamlit run app.py
   ```

6. **Test Inference**
   - Upload sample X-ray
   - Verify prediction and explanations
   - Check access log

---

## 📊 Expected Outputs

### Training Output
```
================================================================================
RAG-Enhanced Vertical Federated Learning for Medical Imaging
================================================================================
Configuration:
  - RAG Enhancement: Enabled
  - Blockchain: Disabled
  - DP Noise (theta): 0.1
  - Dataset size: 0.0125
================================================================================

✓ Initializing model registry and ledger...
  Registry: /path/to/models/registry
  Ledger: /path/to/ledger

✓ RAG module initialized with medical knowledge base
✓ X-ray dataset loaded successfully

--- Epoch [1/3] ---
Time: 45.23s
Val Loss: 0.58 | Val Acc: 65.20%
Test Loss: 0.62 | Test Acc: 63.75%
✓ Model saved: v1.0_round1_20231223_143022

--- Epoch [2/3] ---
...

================================================================================
Training Complete!
✓ RAG-enhanced model trained successfully

📦 Model Registry Summary:
  Total versions: 3
  Latest version: v1.2_round3_20231223_143524
  Best accuracy: 72.50%

📋 Ledger Summary:
  Training entries: 3
  Access entries: 6
  Integrity verified: ✓
================================================================================
```

### Registry JSON Export
```json
{
  "v1.0_round1_20231223_143022": {
    "version_id": "v1.0_round1_20231223_143022",
    "round_num": 1,
    "metrics": {
      "val_accuracy": 65.2,
      "test_accuracy": 63.75,
      "val_loss": 0.58,
      "test_loss": 0.62
    },
    "config": {
      "theta": 0.1,
      "num_clients": 4,
      "use_rag": true,
      "use_blockchain": false
    },
    "model_hash": "abc123...",
    "timestamp": "2023-12-23T14:30:22.123456"
  }
}
```

### Ledger Training Log
```json
{
  "entry_type": "training_round",
  "timestamp": "2023-12-23T14:30:22.123456",
  "round_num": 1,
  "node_metrics": {
    "server": {"val_accuracy": 65.2, "test_accuracy": 63.75},
    "client0": {"status": "trained"},
    "client1": {"status": "trained"}
  },
  "model_hash": "abc123...",
  "blockchain_tx": null,
  "rag_retrieval_count": 0,
  "privacy_budget": 0.1,
  "entry_hash": "def456..."
}
```

---

## 🐛 Troubleshooting

### Issue: "No dataset available"
**Solution**: Run data preparation first
```bash
python prepare_dataset.py --output-dir ./demo_data --create-sample --sample-size 50
```

### Issue: "No models in registry"
**Solution**: Train at least one model
```bash
cd src
python demo_rag_vfl.py --datapath ../demo_data --use-rag --num-epochs 1
```

> **Kaggle → Windows portability**: After training on Kaggle, download the
> `models/registry/` directory and place it in the same location on your Windows
> machine before launching the webapp.  All trained client and server checkpoints
> live under this folder, so copying it is all that is needed to keep model
> versions visible across restarts and across machines.

### Issue: Web UI won't start
**Solution**: Install webapp dependencies
```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py
```

### Issue: "Module not found" errors
**Solution**: Install main dependencies
```bash
pip install torch torchvision numpy pandas streamlit pillow
```

### Issue: Inference fails
**Solution**: Ensure model is trained and image is valid
- Check `models/registry/checkpoints/` has `.pt` files
- Verify image is JPG/PNG format
- Ensure image is chest X-ray (224x224 recommended)

---

## 📚 File Structure

```
MedRAG/
├── src/
│   ├── model_registry.py          # Model versioning system
│   ├── ledger.py                  # Audit ledger system
│   ├── inference.py               # Inference engine
│   ├── demo_rag_vfl.py           # Main training script (modified)
│   ├── models.py                  # VFL model architectures
│   ├── rag_server_model.py       # RAG-enhanced server
│   └── rag_retriever.py          # RAG knowledge base
├── webapp/
│   ├── app.py                    # Main Streamlit app
│   ├── utils.py                  # Shared utilities
│   ├── requirements.txt          # Web dependencies
│   └── pages/
│       ├── 1_🔬_Inference.py    # Inference page
│       ├── 2_📊_Training.py     # Training dashboard
│       ├── 3_📦_Registry.py     # Model registry
│       └── 4_📋_Ledger.py       # Audit ledger
├── models/
│   └── registry/
│       ├── registry.json         # Registry metadata
│       └── checkpoints/          # Model checkpoints (.pt)
├── ledger/
│   ├── training_log.jsonl       # Training logs
│   └── access_log.jsonl         # Access logs
├── run_demo.sh                   # One-command launcher
├── DEMO_GUIDE.md                 # This file
└── README.md                     # Project README

```

---

## 🎬 Demo Script for Presentations

**[0:00-0:30] Introduction**
- "MedRAG combines VFL, blockchain, and RAG for privacy-preserving medical AI"
- "Multiple hospitals collaborate without sharing raw patient data"

**[0:30-2:00] Training Demo**
- Run: `./run_demo.sh`
- Show training progress in terminal
- Explain: "Each hospital trains on local data, only embeddings are shared"

**[2:00-4:00] Web UI Overview**
- Open http://localhost:8501
- Navigate through pages
- Show system status

**[4:00-7:00] Inference Demo**
- Upload sample X-ray
- Show prediction with RAG explanations
- Highlight: "RAG retrieves relevant medical knowledge"

**[7:00-9:00] Audit & Compliance**
- Show training dashboard metrics
- Browse model registry
- Demonstrate ledger integrity verification
- Explain: "Complete audit trail for regulatory compliance"

**[9:00-10:00] Conclusion**
- Summarize key features
- Answer questions

---

## 🔗 References

- [Main README](README.md) - Project overview
- [QUICKSTART](QUICKSTART.md) - Installation guide
- [RAG Documentation](src/RAG_README.md) - RAG component details
- [NOVELTY](NOVELTY.md) - Research contributions

---

## 📞 Support

For issues or questions:
1. Check [Troubleshooting](#-troubleshooting) section
2. Review [Expected Outputs](#-expected-outputs)
3. Verify file structure matches [File Structure](#-file-structure)

---

**Demo prepared for Capstone Review**  
**MedRAG Team - December 2023**
