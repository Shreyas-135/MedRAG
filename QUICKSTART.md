# Quick Start Guide: Running MedRAG

## Title Verification ✅

This implementation **fully supports** your capstone title:

**"A Blockchain-Enabled Vertical Federated Learning Framework for Privacy-Preserving Cross-Hospital Medical Imaging using Verifiable RAG"**

### Component Checklist:
- ✅ **Blockchain-Enabled**: Smart contracts for weight aggregation (Aggregator.sol)
- ✅ **Vertical Federated Learning**: Multi-client collaborative learning with privacy preservation
- ✅ **Privacy-Preserving**: Differential privacy, no raw data sharing between hospitals
- ✅ **Cross-Hospital**: Supports 4 clients (configurable) representing different hospitals
- ✅ **Medical Imaging**: Focused on X-ray image analysis
- ✅ **Verifiable RAG**: Blockchain-verified retrieval operations with cryptographic proofs
- ✅ **X-ray Focus**: COVID-19, pneumonia, and other chest X-ray conditions (excluding MRI/CT)
- ✅ **ResNet/YOLO Support**: ResNet50+VGG19 implemented, YOLO integration ready

## How to Run in Terminal

### Step 1: Install Dependencies

```bash
# Navigate to the project directory
cd /path/to/MedRAG

# Option 1: Use the installation script (recommended)
cd src
# For Linux/Mac (if script exists):
bash installDependencies.sh
# For Windows (if script exists):
installDependencies.cmd

# Option 2: Install manually (always works)
pip install web3 eth-account eth-tester py-solc-x py-evm
pip install numpy torch torchvision Pillow tqdm scikit-learn matplotlib argparse
```

### Step 2: Prepare Your Dataset

**Option A: Automated Dataset Preparation (Recommended)**

Use our automated script to download and prepare datasets:

```bash
# Create sample dataset for quick testing (no download needed)
python prepare_dataset.py --output-dir ./data --create-sample

# Or download real datasets from Kaggle (requires Kaggle API setup)
python prepare_dataset.py --output-dir ./data

# Or use your local datasets
python prepare_dataset.py --output-dir ./data --use-local-data --local-data-dir ./raw_datasets
```

See **[DATASET_PREPARATION.md](DATASET_PREPARATION.md)** for complete documentation.

**Option B: Manual Dataset Preparation**

Download and extract your X-ray dataset. The system expects this structure:

```
your_dataset_path/
├── SplitCovid19/
│   ├── client0/
│   │   ├── train/
│   │   │   ├── covid/      # Class 0 or 1
│   │   │   └── normal/     # Class 0 or 1
│   │   └── test/
│   │       ├── covid/
│   │       └── normal/
│   ├── client1/
│   ├── client2/
│   └── client3/
```

**Note**: If you have a different X-ray dataset (ChestX-ray14, MIMIC-CXR, etc.), you can modify the dataset loading in `demo_rag_vfl.py`.

### Step 3: Run the Code

#### Option A: RAG-Enhanced VFL (Recommended for Full Title Support)

```bash
cd src

# Basic RAG-enhanced training
python demo_rag_vfl.py --datapath /path/to/your/dataset --use-rag

# With blockchain verification (full title implementation)
python demo_rag_vfl.py --datapath /path/to/your/dataset --use-rag --withblockchain

# Custom configuration
python demo_rag_vfl.py \
    --datapath /path/to/your/dataset \
    --use-rag \
    --withblockchain \
    --theta 0.15 \
    --datasize 0.5 \
    --num-epochs 10
```

#### Option B: Standard VFL (Without RAG)

```bash
cd src

# Without blockchain
python demo.py --datapath /path/to/your/dataset --datasize 0.5 --theta 0.1

# With blockchain
python demo.py --datapath /path/to/your/dataset --datasize 0.5 --theta 0.1 --withblockchain
```

#### Option C: VFL Without Blockchain (Basic)

```bash
cd src
python vfl_without_blockchain.py /path/to/your/dataset --theta 0.15
```

#### Option D: Breast Cancer Dataset Example

```bash
cd src
python main_breast_cancer.py --num_clients 5 --theta 0.1 --quant_bin 8
```

### Step 4: Verify RAG and Blockchain Integration

The RAG-enhanced version will display:

```
================================================================================
RAG-Enhanced Vertical Federated Learning for Medical Imaging
A Blockchain-Enabled Framework for Privacy-Preserving Cross-Hospital X-ray Analysis
================================================================================
Configuration:
  - RAG Enhancement: Enabled
  - Blockchain: Enabled
  - DP Noise (theta): 0.1
  - Dataset size: 1.0
================================================================================
✓ Blockchain integration initialized
✓ RAG module initialized with medical knowledge base
  Knowledge base hash: <hash>...
✓ X-ray dataset loaded successfully
```

## Command Line Arguments Explained

### Common Arguments (all scripts)

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--datapath` | Path to dataset directory | `./` | `--datapath /data/xrays` |
| `--theta` | DP noise parameter (0-0.25) | `0.1` | `--theta 0.15` |
| `--datasize` | Dataset portion to use | `1.0` | `--datasize 0.5` |

### RAG-Specific Arguments (demo_rag_vfl.py)

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--use-rag` | Enable RAG enhancement | `False` | `--use-rag` |
| `--withblockchain` | Enable blockchain | `False` | `--withblockchain` |
| `--num-epochs` | Training epochs | `5` | `--num-epochs 10` |

### Breast Cancer Arguments (main_breast_cancer.py)

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--num_clients` | Number of clients | `5` | `--num_clients 4` |
| `--quant_bin` | Quantization buckets | `0` | `--quant_bin 8` |
| `--withblockchain` | Enable blockchain | `False` | `--withblockchain` |

## Testing Without Dataset

If you don't have the dataset yet, you can still test the architecture:

```bash
cd src

# This will initialize the RAG architecture and show it's working
# (will skip training but demonstrate the components)
python demo_rag_vfl.py --use-rag --datapath ./dummy_path
```

## Troubleshooting

### Issue 1: "No module named 'torch'"
**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

### Issue 2: "No module named 'web3'"
**Solution**: Install blockchain dependencies
```bash
pip install web3 eth-account eth-tester py-solc-x py-evm
```

### Issue 3: "Dataset not found"
**Solution**: Provide correct dataset path or create dummy structure
```bash
mkdir -p dummy_dataset/SplitCovid19/client0/train/covid
mkdir -p dummy_dataset/SplitCovid19/client0/test/covid
# Then use: --datapath dummy_dataset
```

### Issue 4: "Blockchain integration not available"
**Solution**: Make sure you're in the `src` directory and blockchain dependencies are installed
```bash
cd src
pip install web3 eth-account eth-tester py-solc-x py-evm
```

### Issue 5: DataLoader multiprocessing errors / `AssertionError: can only test a child process` on shutdown (GitHub Codespaces / Python 3.12)
**Solution**: `demo_rag_vfl_with_zip.py` already defaults to `num_workers=0` for all DataLoaders, which avoids this issue. If you see this error in other scripts, set `num_workers=0` in any `DataLoader(...)` calls. Codespaces and Python 3.12 are more stable with `num_workers=0` because multiprocessing worker teardown behaves differently in those environments.

## Output Examples

### Successful RAG-Enhanced Training Output:

```
================================================================================
RAG-Enhanced Vertical Federated Learning for Medical Imaging
A Blockchain-Enabled Framework for Privacy-Preserving Cross-Hospital X-ray Analysis
================================================================================
Configuration:
  - RAG Enhancement: Enabled
  - Blockchain: Enabled
  - DP Noise (theta): 0.1
  - Dataset size: 1.0
================================================================================
✓ Blockchain integration initialized
✓ RAG module initialized with medical knowledge base
  Knowledge base hash: a7b3c5d9e2f1...
✓ X-ray dataset loaded successfully

Initial Evaluation:
  Test loss: 0.69 | Test accuracy: 52.50%

Starting training for 5 epochs...

--- Epoch [1/5] ---
Time: 45.32s
Val Loss: 0.58 | Val Acc: 65.20%
Test Loss: 0.62 | Test Acc: 63.75%
RAG KB Hash: a7b3c5d9e2f1...

--- Epoch [2/5] ---
...

================================================================================
Training Complete!
✓ RAG-enhanced model trained successfully
✓ Knowledge base integrity verified: a7b3c5d9e2f1...
✓ Blockchain aggregation verified
================================================================================
```

## Next Steps for Your Capstone

1. **Collect/Prepare X-ray Dataset**:
   - Download COVID-19 X-ray datasets (e.g., from Kaggle, COVID-19 Radiography Database)
   - Split vertically among 4 clients (each client gets a quadrant of each image)

2. **Customize Knowledge Base**:
   - Edit `rag_retriever.py` to add domain-specific medical knowledge
   - Populate with real radiologist findings and diagnoses

3. **Experiment with Architectures**:
   - Current: ResNet50 + VGG19
   - Optional: Integrate YOLO (see RAG_README.md for instructions)

4. **Evaluate and Document**:
   - Run experiments with different configurations
   - Document accuracy improvements with RAG
   - Demonstrate blockchain verification

5. **Novelty Aspects to Highlight**:
   - Combination of VFL + Blockchain + RAG (unique integration)
   - Verifiable RAG with blockchain (novel for medical imaging)
   - Privacy-preserving retrieval in federated settings
   - Cross-hospital collaboration without data sharing

## Project Structure

```
MedRAG/
├── src/
│   ├── demo_rag_vfl.py          # Main RAG-enhanced demo (USE THIS)
│   ├── rag_retriever.py         # RAG knowledge base & retrieval
│   ├── rag_server_model.py      # RAG-enhanced server model
│   ├── RAG_README.md            # Detailed RAG documentation
│   ├── demo.py                  # Standard VFL demo
│   ├── vfl_without_blockchain.py # Basic VFL
│   ├── main_breast_cancer.py    # Breast cancer example
│   ├── models.py                # Client and server models
│   ├── Blockchain_and_VFL_Integration.py  # Blockchain integration
│   ├── Aggregator.sol           # Smart contract
│   └── installDependencies.sh   # Dependency installer
├── tests/
│   └── test_aggregator.py       # Smart contract tests
└── README.md                    # Main documentation
```

## Key Files for Your Capstone Demonstration

1. **Main Demo**: `src/demo_rag_vfl.py` - Shows full title implementation
2. **RAG Components**: `src/rag_retriever.py` and `src/rag_server_model.py`
3. **Blockchain**: `src/Aggregator.sol` and `src/Blockchain_and_VFL_Integration.py`
4. **Models**: `src/models.py` - ResNet50 + VGG19 implementation
5. **Documentation**: `src/RAG_README.md` - Detailed technical docs

## Citation for Your Capstone

```bibtex
@software{medrag_vfl_blockchain,
  title={A Blockchain-Enabled Vertical Federated Learning Framework for 
         Privacy-Preserving Cross-Hospital Medical Imaging using Verifiable RAG},
  author={[Your Team Name]},  # Replace with your team members' names
  year={2024},
  url={https://github.com/Shreyas-135/MedRAG}  # Verify this matches your repository
}
```

## Support

For issues or questions:
1. Check this guide and RAG_README.md
2. Verify dependencies are installed
3. Test with the example commands above
4. Check that you're running from the `src` directory
