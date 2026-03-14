# MedRAG: A Blockchain-Enabled Vertical Federated Learning Framework for Privacy-Preserving Cross-Hospital Medical Imaging using Verifiable RAG

## Overview

This project implements a comprehensive framework that combines:
- **Vertical Federated Learning (VFL)** for privacy-preserving collaborative learning across hospitals
- **Blockchain** for transparent and immutable weight aggregation
- **Retrieval-Augmented Generation (RAG)** for enhanced medical image analysis with verifiable knowledge
- **Differential Privacy** for protecting model weight updates

## Key Features

✨ **RAG-Enhanced Medical Imaging**: Retrieval-augmented generation improves diagnostic accuracy by leveraging curated medical knowledge
🔐 **Privacy-Preserving**: Hospitals collaborate without sharing raw patient data
⛓️ **Blockchain Verification**: Transparent and immutable record of model updates and RAG retrievals
🏥 **Multi-Hospital Support**: Designed for cross-hospital collaboration
📊 **X-ray Focus**: Optimized for X-ray image analysis (COVID-19, pneumonia, normal, etc.)
🎯 **Demo Infrastructure**: Complete web UI for inference, training monitoring, and audit trails

## 🚀 Quick Demo (NEW!)

Experience the complete MedRAG system with our interactive web interface:

```bash
# One-command demo setup and launch
./run_demo.sh
```

This launches:
- 🔬 **Inference UI**: Upload X-rays for AI-powered COVID-19 detection with RAG explanations
- 📊 **Training Dashboard**: Real-time metrics and performance visualization
- 📦 **Model Registry**: Browse and compare model versions
- 📋 **Audit Ledger**: Complete training and access logs with integrity verification

**Demo Guide**: See [DEMO_GUIDE.md](DEMO_GUIDE.md) for detailed walkthrough

**Web Interface**: After running the demo, navigate to http://localhost:8501

## 🔗 Blockchain Integration

MedRAG includes comprehensive blockchain features for transparent and verifiable federated learning:

### 🌐 New Web UI Features

- **🏥 Hospital Network Overview**: Visualize 4-hospital network topology with real-time status
- **⛓️ Blockchain Explorer**: View blocks, transactions, and weight aggregations on-chain
- **💰 Wallet Integration**: MetaMask connection for transaction signing (demo mode)
- **📊 Enhanced Training Dashboard**: Compare VFL, VFL+Blockchain, VFL+RAG, and Full System

### 🛠️ Ganache Setup (Optional)

To view **real blockchain data** instead of mock data:

1. **Install Ganache**: Download from [trufflesuite.com/ganache](https://trufflesuite.com/ganache/)
2. **Start Ganache**: Run on port 7545 with network ID 1337
3. **Run with Blockchain**:
   ```bash
   cd src
   python demo_rag_vfl.py --datapath ../demo_data --use-rag --withblockchain
   ```
4. **View in UI**: Navigate to Blockchain Explorer page to see live transactions

**📖 Complete Setup Guide**: See [GANACHE_SETUP.md](GANACHE_SETUP.md) for detailed instructions including MetaMask integration.

### Benefits of Blockchain Integration

- ✅ **Transparency**: All weight aggregations visible to all hospitals
- ✅ **Immutability**: Complete audit trail that cannot be altered
- ✅ **Trustless**: No central authority needed for coordination
- ✅ **Verifiable**: Cryptographic proof of all model updates
- ✅ **Resilient**: Distributed system with no single point of failure

# Motivation
Hospitals have huge amounts of data that they might not like to share openly with other hospitals. However, hospitals benefit from having a Machine Learning model that has been trained on data from other hospitals for better insights on their own data. Vertical Federated Learning is a good fit for this use case. Blockchain is used for transparency and immutability of the global weight updates. Local Differential Privacy provides protection for model weight updates that are published on the Blockchain.

**NEW**: Retrieval-Augmented Generation (RAG) enhances the framework by allowing the model to retrieve relevant medical knowledge during inference, improving diagnostic accuracy and providing explainable results. The RAG component is verifiable through blockchain, ensuring transparency and trust in the retrieved medical knowledge.

# How to run

## 📚 **Complete Guides**

- **[DEMO_GUIDE.md](DEMO_GUIDE.md)** - Complete demo walkthrough with web UI
- **[INTERVIEW_DEMO_GUIDE.md](INTERVIEW_DEMO_GUIDE.md)** - Quick interview demo setup
- **[QUICKSTART.md](QUICKSTART.md)** - Comprehensive guide with terminal commands and examples
- **[NOVELTY.md](NOVELTY.md)** - Detailed analysis of novel contributions and research value
- **[src/RAG_README.md](src/RAG_README.md)** - Technical RAG component documentation

## Demo Infrastructure (NEW!)

### Model Registry System

Track and manage model versions with complete provenance:

```python
from model_registry import ModelRegistry

registry = ModelRegistry()
version_id = registry.save_model(model, round_num=5, 
                                 metrics={'accuracy': 0.85}, 
                                 config={'use_rag': True})
best_model = registry.get_best_model('accuracy')
```

### Comprehensive Ledger

Immutable audit trail for compliance and transparency:

```python
from ledger import Ledger

ledger = Ledger()
ledger.log_training_round(round_num=1, node_metrics={...}, 
                         model_hash="abc123", privacy_budget=0.1)
is_valid = ledger.verify_integrity('training')  # Verify hash chain
```

### Inference Engine

Single-image prediction with RAG explanations:

```python
from inference import load_inference_model

inference = load_inference_model(use_rag=True)
result = inference.predict('xray.jpg')
# Returns: prediction, confidence, RAG explanations, guidelines
```

## Quick Start with RAG

```bash
# Run RAG-enhanced VFL (recommended)
cd src
python demo_rag_vfl.py --datapath /path/to/xray/dataset --use-rag

# Run with blockchain integration (implements complete framework: VFL + Blockchain + Verifiable RAG)
python demo_rag_vfl.py --datapath /path/to/xray/dataset --use-rag --withblockchain

# Custom configuration
python demo_rag_vfl.py --datapath /path/to/xray/dataset --use-rag --withblockchain \
    --theta 0.15 --datasize 0.5 --num-epochs 10
```

## Standard VFL (without RAG)
[src/README.md](src/README.md)

# Implementation
Our project uses Vertical Federated Learning for binary classification of medical image data from different hospitals. A Smart Contract is used for aggregation (summation) of local training results (embedding) to train the global model. The embedding sum is stored on the Blockchain and sent to a central server. The central server trains its model on the embedding sum sent by the Smart Contract and sends the gradient back to each client.

# Model Architecture 
![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/images/Model%20Architecture.png)

# Sequence Diagram
![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/images/Sequence_Diagram.png)

# Training Workflow 
![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/images/Client%20Model.png)

1. At first, two CNN pre-trained models, i.e., ResNet50, and VGG19, with the pre-trained weights will be adopted for the client model.
2. We will use this model without their classification layers because we want to use these for feature extraction part only.
3. All the extracted features will be combined into a single fusion vector (embedding) using a concatenate layer.
4. The embeddings represent high-level functionality such as sharpening, textures, roundness, and compactness of the CXR images.
5. Finally, the embeddings are summed and then fed into the central server for the training and classification purpose.

# Vertical Federated Learning Algorithm with RAG

Below is a description of our Vertical Federated Learning algorithm enhanced with RAG.

In each training round:
1. A minibatch is randomly chosen for training. The IDs of the chosen samples are shared among server and clients
2. Each client generates embeddings using their local model and private data.
3. Each client adds differential privacy noise to their embeddings.
4. Each client sends their noisy embeddings to the smart contract for aggregation.
5. Smart contract sums the noisy embedding and sends to the server.
6. **[RAG Enhancement]** Server retrieves relevant medical knowledge from the knowledge base based on the aggregated embeddings.
7. **[RAG Enhancement]** Retrieved medical context is integrated with embeddings using multi-head attention.
8. Server calculates the gradient w.r.t the embedding sum (and RAG-enhanced features) and sends to parties.
9. Server calculates the gradient w.r.t the global parameters and updates the global parameters.
10. Each client calculates the gradient w.r.t their local parameters using the chain rule and updates their local parameters.
11. **[RAG Verification]** RAG retrieval operations are logged with cryptographic hashes for blockchain verification.


# Blockchain Component

**Implementation:** A smart contract written in Solidity capable of receiving client weight updates, aggregating them and sending the aggregation to a global model. 

**The smart contract will:**

1. Allow for clients to send their noisy weight updates

2. Receive the weights and sum them together

3. Allow for a global model to receive the aggregated weights from the clients

The smart contract will not perform any training, since it will have high gas costs

**Rationale:** Allows for multiple hospitals to interface with the model without the need for a centralized authority. Blockchain will provide transparency and verifiability, and local differential privacy mechanisms will ensure privacy. 

# Dataset

We use an image dataset with binary classification that predicts if a medical image has Covid-19 or not. The dataset is vertically splitted among 4 parties so that each party holds a quadrant of each image. The split dataset can be retrieved using the Google Drive [link](https://drive.google.com/file/d/1LUGy0TA03C-wcLBk8YGDeVJ42u2yHmY_/view?usp=sharing).

## Automated Dataset Preparation

**NEW**: We now provide an automated script to download and prepare X-ray datasets! See **[DATASET_PREPARATION.md](DATASET_PREPARATION.md)** for the complete guide.

```bash
# Quick start - create sample dataset for testing
python prepare_dataset.py --output-dir ./data --create-sample

# Or download real datasets from Kaggle
python prepare_dataset.py --output-dir ./data
```

The script automatically:
- Downloads COVID-19, Pneumonia, and Tuberculosis X-ray datasets from Kaggle
- Organizes images into the required directory structure
- Splits data into train/test subsets
- Distributes images across 4 federated learning clients

## X-ray Dataset Support

The RAG-enhanced framework is designed to work with various X-ray datasets including:
- **COVID-19 Detection**: Chest X-rays for COVID-19 vs. normal classification
- **Pneumonia Detection**: Bacterial and viral pneumonia identification
- **General Chest X-ray Analysis**: Multiple pathology detection (pleural effusion, cardiomegaly, etc.)
- **Multi-Class X-ray Datasets**: ChestX-ray14, MIMIC-CXR, and similar datasets

The framework focuses on X-ray imaging (excluding MRI and CT scans) and can be adapted to work with different X-ray datasets by modifying the data loading pipeline.

### NEW: Using Your Own ZIP Dataset

You can now use your own X-ray datasets in ZIP format! The framework automatically extracts, categorizes, and distributes images across federated hospitals.

```bash
# Step 1: Extract and organize your ZIP file
python src/load_zip_dataset.py --zip-file my_xrays.zip --output-dir ./data

# Step 2: Train with your data (supports YOLO models!)
python src/demo_rag_vfl_with_zip.py --datapath ./data --model-type yolo5 --use-rag
```

**See [docs/USING_YOUR_XRAY_ZIP.md](docs/USING_YOUR_XRAY_ZIP.md) for the complete guide.**

Features:
- ✅ Automatic image categorization from filenames/folders
- ✅ Even distribution across Hospital A, B, C, D
- ✅ Supports YOLO (YOLOv5, YOLOv8) and hybrid architectures
- ✅ Binary and multi-class classification
- ✅ Progress tracking and statistics

# RAG Component

## Overview

The Retrieval-Augmented Generation (RAG) component enhances the VFL framework by incorporating a medical knowledge base that stores curated X-ray findings and diagnoses. During inference, the system retrieves relevant medical knowledge based on the patient's X-ray features and uses this context to improve diagnostic accuracy.

## Key Components

1. **Medical Knowledge Base** (`rag_retriever.py`)
   - Stores medical findings, diagnoses, and clinical observations
   - Uses embedding-based similarity search
   - Supports privacy-preserving retrieval
   - Blockchain-verifiable with cryptographic hashing

2. **RAG-Enhanced Server Model** (`rag_server_model.py`)
   - Integrates RAG with VFL server
   - Multi-head attention for context integration
   - Enhanced classification with retrieved knowledge
   - Verifiable knowledge base integrity

3. **Blockchain Verification**
   - Knowledge base hash stored on blockchain
   - Retrieval operations logged with proofs
   - Transparent and auditable RAG operations
   - Hospitals can verify knowledge integrity

## Usage Example

```python
# Initialize RAG-enhanced VFL system
from rag_server_model import RAGEnhancedServerModel

server_model = RAGEnhancedServerModel(
    embedding_dim=64,
    num_classes=2,
    use_rag=True
)

# Populate knowledge base
knowledge_entries = [
    {
        'text': 'Bilateral ground-glass opacities consistent with COVID-19',
        'embedding': covid_features,
        'condition': 'covid',
        'severity': 'moderate'
    },
    # More entries...
]
server_model.update_knowledge_base(knowledge_entries)

# Verify knowledge base integrity
kb_hash = server_model.verify_rag_integrity()
print(f"Knowledge base hash: {kb_hash}")
```

## Benefits

- **Improved Accuracy**: Leverages curated medical knowledge for better predictions
- **Explainability**: Retrieved findings provide context for diagnoses
- **Consistency**: Shared knowledge base ensures consistent medical interpretations
- **Verifiability**: Blockchain ensures transparent and auditable RAG operations
- **Privacy-Preserving**: Works within federated learning framework

## Architecture Support

The RAG framework supports multiple model architectures for X-ray analysis:

### Traditional CNN Models
- **ResNet50 + VGG19**: Current default implementation for feature extraction
- **ResNet50 + DenseNet169 + VGG19**: Three-way feature fusion

### YOLO Models (NEW!)
- **YOLOv5**: Object detection-based feature extraction
- **YOLOv8**: Latest YOLO architecture for X-ray analysis
- **Hybrid ResNet + YOLO**: Combines CNN and object detection features

### Vision Transformers
- **ViT (Vision Transformer)**: Attention-based architecture
- **Hybrid CNN + ViT**: Best of both worlds

All architectures produce compatible 64-dimensional embeddings for the VFL framework.

**Model Comparison:**
```bash
python src/demo_rag_vfl_with_zip.py --compare-models --model-type yolo5
```

For more details, see [src/RAG_README.md](src/RAG_README.md).

## 🚀 Advanced RAG Configuration (NEW!)

### LangChain + ChromaDB + Gemini Integration

The framework now supports enterprise-grade RAG with:
- **ChromaDB** for persistent vector database storage  
- **Google Gemini** for natural language medical explanations
- **LangChain** for orchestrating retrieval and generation
- **Flower** framework for federated learning orchestration

### Quick Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up Gemini API key:**
```bash
export GEMINI_API_KEY="your-api-key-here"
# Get your key from: https://makersuite.google.com/app/apikey
```

3. **Initialize ChromaDB with medical knowledge:**
```bash
python scripts/setup_rag_components.py
```

### Usage Examples

**Training with LangChain RAG + Flower:**
```bash
python src/demo_flower_vfl.py \
  --datapath ./demo_data \
  --use-flower \
  --use-langchain \
  --num-rounds 10 \
  --withblockchain
```

**Inference with LLM Explanations:**
```python
from inference import load_inference_model

# Load model with LangChain RAG
engine = load_inference_model(use_rag=True, use_langchain=True)

# Get prediction with Gemini explanation
result = engine.predict('xray.jpg', use_llm_explanation=True)

print(result['rag_explanation'])  # Gemini-generated clinical explanation
print(result['citations'])         # Retrieved medical literature references
```

**Configuration:** See `config/langchain_rag_example.yaml` for all options.

**Documentation:** 
- Complete guide: [docs/LANGCHAIN_RAG_GUIDE.md](docs/LANGCHAIN_RAG_GUIDE.md)
- Configuration reference: [src/config/rag_config.py](src/config/rag_config.py)



# Experiment Results

We conducted the experiment on the AiMOS cluster and recorded the results as follow. The line graphs show the test accuracy by epoch with and without blockchain for experiment on the full Covid-19 dataset. The bar graph show the runtime took to train the model with three different dataset sizes.

![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/images/accuracy_with_bc.png)

![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/images/accuracy_without_bc.png)

![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/images/runtime.png)
