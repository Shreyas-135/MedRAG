# LangChain RAG Architecture Guide

This document provides a comprehensive guide to the LangChain-based Retrieval-Augmented Generation (RAG) system integrated with the MedRAG federated learning framework.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [ChromaDB Schema](#chromadb-schema)
5. [Gemini Integration](#gemini-integration)
6. [Flower Federation](#flower-federation)
7. [Blockchain Integration](#blockchain-integration)
8. [Usage Examples](#usage-examples)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)

## Overview

The advanced RAG system combines:
- **ChromaDB**: Vector database for persistent medical knowledge storage
- **Google Gemini**: LLM for generating clinical explanations
- **LangChain**: Orchestration framework for RAG pipeline
- **Flower**: Federated learning framework for distributed training
- **Blockchain**: Immutable audit trail for explanations and retrievals

### Key Benefits

✅ **Explainable AI**: Clinically-relevant explanations from Gemini LLM  
✅ **Persistent Storage**: ChromaDB ensures knowledge base persistence  
✅ **Verifiable**: Blockchain hashes for audit trail  
✅ **Scalable**: Flower enables true federated orchestration  
✅ **Backward Compatible**: All existing features still work

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       MedRAG System                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  Hospital A    │  │  Hospital B    │  │  Hospital C    │   │
│  │  (Client)      │  │  (Client)      │  │  (Client)      │   │
│  │  - ResNet      │  │  - VGG         │  │  - DenseNet    │   │
│  │  - Local Data  │  │  - Local Data  │  │  - Local Data  │   │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘   │
│           │                   │                   │             │
│           └───────────────────┼───────────────────┘             │
│                               │                                 │
│                    ┌──────────▼──────────┐                      │
│                    │  Flower FL Server   │                      │
│                    │  - Aggregation      │                      │
│                    │  - Coordination     │                      │
│                    └──────────┬──────────┘                      │
│                               │                                 │
│                    ┌──────────▼──────────┐                      │
│                    │  RAG-Enhanced       │                      │
│                    │  Server Model       │                      │
│                    │  ┌──────────────┐   │                      │
│                    │  │ LangChain    │   │                      │
│                    │  │ RAG Pipeline │   │                      │
│                    │  └──────┬───────┘   │                      │
│                    └─────────┼───────────┘                      │
│           ┌──────────────────┼──────────────────┐               │
│           │                  │                  │               │
│  ┌────────▼────────┐ ┌───────▼──────┐ ┌────────▼────────┐     │
│  │ ChromaDB        │ │ Gemini API   │ │ Blockchain      │     │
│  │ - Medical KB    │ │ - LLM Gen.   │ │ - Audit Trail   │     │
│  │ - Embeddings    │ │ - Explain.   │ │ - Verification  │     │
│  └─────────────────┘ └──────────────┘ └─────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. ChromaDBMedicalKnowledgeBase

Persistent vector database for medical knowledge.

**Features:**
- Cosine similarity search
- Metadata filtering (condition, severity, location)
- Batch operations
- Deterministic hashing for blockchain verification

**Example:**
```python
from langchain_rag import ChromaDBMedicalKnowledgeBase

kb = ChromaDBMedicalKnowledgeBase()

# Add medical finding
kb.add_entry(
    text="Bilateral ground-glass opacities consistent with COVID-19",
    embedding=image_features,  # 64-dim vector
    metadata={
        'condition': 'covid',
        'severity': 'moderate',
        'findings': ['ground_glass_opacity', 'bilateral']
    }
)

# Search for similar cases
results = kb.search(query_embedding, top_k=5)
for result in results:
    print(f"Similarity: {result['similarity']:.2f}")
    print(f"Text: {result['text']}")
    print(f"Condition: {result['metadata']['condition']}")
```

### 2. GeminiMedicalExplainer

Generates natural language explanations using Google Gemini.

**Features:**
- Clinical-style explanations
- Citation generation
- Configurable temperature and length
- Fallback to simple explanations if API fails

**Example:**
```python
from langchain_rag import GeminiMedicalExplainer

explainer = GeminiMedicalExplainer(api_key="your-key")

result = explainer.generate_explanation(
    prediction="COVID-19",
    confidence=0.87,
    retrieved_cases=similar_cases,
    include_citations=True
)

print(result['explanation'])
# Output: "Based on the analysis, the model predicts COVID-19 
# with 87% confidence. The chest X-ray shows bilateral 
# ground-glass opacities predominantly in peripheral and 
# lower lung zones, consistent with viral pneumonia..."

print(result['citations'])
# Output: [{'case_number': 1, 'similarity': 0.95, ...}, ...]
```

### 3. LangChainRAGPipeline

Orchestrates retrieval and generation workflow.

**Workflow:**
1. Retrieve similar medical cases from ChromaDB
2. Generate LLM explanation using Gemini
3. Compute blockchain hashes
4. Return formatted response with citations

**Example:**
```python
from langchain_rag import LangChainRAGPipeline

pipeline = LangChainRAGPipeline()

result = pipeline.query(
    embedding=image_embedding,
    prediction="COVID-19",
    confidence=0.87,
    top_k=3,
    generate_explanation=True
)

print(result['rag_explanation'])       # LLM explanation
print(result['retrieved_cases'])       # Similar cases
print(result['citations'])             # Literature refs
print(result['explanation_hash'])      # For blockchain
```

### 4. FlowerMedicalClient

Flower client for federated learning.

**Features:**
- Differential privacy for gradients
- Local training with configurable epochs
- Compatible with PyTorch models
- Automatic parameter serialization

**Example:**
```python
from flower_vfl import FlowerMedicalClient

client = FlowerMedicalClient(
    client_id='Hospital_A',
    model=resnet_model,
    train_loader=train_data,
    apply_dp=True,
    dp_noise_multiplier=0.1
)

# Client automatically handles fit/evaluate
```

### 5. FlowerVFLStrategy

Custom aggregation strategy for VFL.

**Features:**
- FedAvg-based aggregation
- Blockchain logging integration
- RAG-enhanced server model support
- Configurable client sampling

**Example:**
```python
from flower_vfl import FlowerVFLStrategy

strategy = FlowerVFLStrategy(
    server_model=rag_server,
    blockchain_integrator=blockchain,
    rag_pipeline=pipeline,
    min_fit_clients=4,
    min_available_clients=4
)
```

## ChromaDB Schema

### Collection Structure

**Collection Name:** `medical_xray_findings`

**Schema:**
```python
{
    'id': str,              # MD5 hash of text
    'document': str,        # Medical finding description
    'embedding': [float],   # 64-dimensional vector
    'metadata': {
        'condition': str,     # e.g., 'covid', 'normal', 'pneumonia'
        'severity': str,      # e.g., 'mild', 'moderate', 'severe', 'none'
        'findings': [str],    # e.g., ['ground_glass_opacity', 'bilateral']
        'location': str       # e.g., 'bilateral_peripheral', 'right_lower'
    }
}
```

### Indexing

ChromaDB uses **HNSW (Hierarchical Navigable Small World)** indexing for fast similarity search:
- Index type: Cosine similarity
- Approximate nearest neighbors
- Sub-100ms queries for 10k+ entries

### Sample Entry

```python
{
    'id': '3f2d8a9b...',
    'document': 'Bilateral ground-glass opacities in peripheral and lower lung zones, consistent with COVID-19 pneumonia',
    'embedding': [0.23, -0.45, 0.67, ...],  # 64 dims
    'metadata': {
        'condition': 'covid',
        'severity': 'moderate',
        'findings': ['ground_glass_opacity', 'bilateral_involvement', 'peripheral_distribution'],
        'location': 'bilateral_peripheral'
    }
}
```

## Gemini Integration

### Prompt Template

The system uses a structured prompt for consistent clinical explanations:

```
You are a medical AI assistant analyzing chest X-ray images.
Based on the model prediction and similar cases from the medical 
knowledge base, provide a clear, clinically-relevant explanation.

**Model Prediction**: COVID-19
**Confidence**: 87.0%

**Similar Cases from Medical Knowledge Base**:
Case 1 (similarity: 95.2%):
  Condition: covid
  Severity: moderate
  Description: Bilateral ground-glass opacities...

Please provide:
1. A brief clinical interpretation of the prediction
2. Key radiological findings that support this diagnosis
3. Clinical significance and recommendations
4. References to the similar cases used in this analysis

Keep the explanation concise, professional, and suitable for 
clinical use. Limit to 3-4 paragraphs.
```

### Response Format

Gemini returns structured JSON:

```python
{
    'explanation': str,        # Full clinical explanation
    'citations': [             # Retrieved case references
        {
            'case_number': int,
            'similarity': float,
            'condition': str,
            'text_snippet': str
        }
    ],
    'confidence': float,
    'prediction': str,
    'num_retrieved_cases': int,
    'model': str               # 'gemini-1.5-flash'
}
```

### Rate Limiting

Gemini API has rate limits:
- **Free tier**: 60 requests/minute
- **Paid tier**: Higher limits available

The system handles rate limits gracefully:
- Automatic retry with exponential backoff
- Falls back to simple RAG if API unavailable
- Caches responses for repeated queries (optional)

## Flower Federation

### Training Workflow

```
1. Server starts and waits for clients
2. Clients connect (Hospital A, B, C, D)
3. For each round (1 to num_rounds):
   a. Server sends global model to clients
   b. Clients train locally with differential privacy
   c. Clients send embeddings back to server
   d. Server aggregates using FedAvg
   e. Server logs to blockchain (optional)
   f. Server evaluates aggregated model
4. Server saves final model
```

### Configuration

**Server:**
```python
from flower_vfl import start_flower_server, FlowerVFLStrategy

strategy = FlowerVFLStrategy(
    min_fit_clients=4,
    min_available_clients=4,
    server_model=rag_server,
    blockchain_integrator=blockchain
)

start_flower_server(
    server_address='[::]:8080',
    num_rounds=10,
    strategy=strategy
)
```

**Client:**
```python
from flower_vfl import FlowerMedicalClient, start_flower_client

client = FlowerMedicalClient(
    client_id='Hospital_A',
    model=client_model,
    train_loader=train_data,
    apply_dp=True
)

start_flower_client(
    server_address='localhost:8080',
    client=client.to_client()
)
```

## Blockchain Integration

### Hash Generation

The system generates cryptographic hashes at multiple levels:

**1. Knowledge Base Hash:**
```python
kb_hash = kb.get_hash()  # SHA-256 of all entries
# Use for: Verify knowledge base integrity
```

**2. Explanation Hash:**
```python
explanation_hash = pipeline._compute_explanation_hash(result)
# Use for: Audit trail of each prediction
```

**3. Retrieval Log Hash:**
```python
retrieval_hash = hash(query_embedding + results)
# Use for: Track which cases were retrieved
```

### Audit Trail

Each RAG query creates an audit record:

```python
{
    'timestamp': '2024-01-27T10:30:00Z',
    'query_hash': 'abc123...',
    'results_hash': 'def456...',
    'kb_hash': 'ghi789...',
    'explanation_hash': 'jkl012...',
    'prediction': 'COVID-19',
    'confidence': 0.87,
    'num_cases_retrieved': 3
}
```

This can be stored on blockchain for:
- Immutable record of all predictions
- Verification of knowledge base state
- Compliance and auditing

## Usage Examples

### Example 1: Basic RAG Query

```python
from langchain_rag import LangChainRAGPipeline, ChromaDBMedicalKnowledgeBase
import numpy as np

# Initialize
kb = ChromaDBMedicalKnowledgeBase()
pipeline = LangChainRAGPipeline(knowledge_base=kb)

# Get image embedding (from your model)
image_embedding = extract_features(xray_image)  # Returns np.array(64,)

# Query RAG pipeline
result = pipeline.query(
    embedding=image_embedding,
    prediction='COVID-19',
    confidence=0.87,
    top_k=3,
    generate_explanation=True
)

# Use results
print(f"Prediction: {result['prediction']}")
print(f"Explanation: {result['rag_explanation']}")
print(f"Retrieved {len(result['retrieved_cases'])} similar cases")
```

### Example 2: Federated Training with Flower

```python
# On server machine
from flower_vfl import start_flower_server, FlowerVFLStrategy
from rag_server_model import RAGEnhancedServerModel

server_model = RAGEnhancedServerModel(use_rag=True, use_langchain=True)
strategy = FlowerVFLStrategy(server_model=server_model)

start_flower_server(
    server_address='[::]:8080',
    num_rounds=10,
    strategy=strategy
)

# On each client machine (Hospital A, B, C, D)
from flower_vfl import FlowerMedicalClient, start_flower_client
from models import ClientModel2Layers

client_model = ClientModel2Layers()
client = FlowerMedicalClient(
    client_id='Hospital_A',
    model=client_model,
    train_loader=hospital_a_data
)

start_flower_client('server_ip:8080', client)
```

### Example 3: Inference with Full Explanation

```python
from inference import load_inference_model

# Load model with all features
engine = load_inference_model(
    checkpoint_path='./checkpoints/model.pth',
    use_rag=True,
    use_langchain=True
)

# Get prediction with full explanation
result = engine.predict(
    image_path='patient_xray.jpg',
    return_explanations=True,
    use_llm_explanation=True
)

# Display results
print(f"Diagnosis: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"\nClinical Explanation:\n{result['rag_explanation']}")
print(f"\nRetrieved Cases:")
for i, case in enumerate(result['retrieved_cases'], 1):
    print(f"  {i}. {case['text'][:80]}... (similarity: {case['similarity']:.2f})")
```

## Configuration

### Environment Variables

```bash
# Required for Gemini
export GEMINI_API_KEY="your-api-key-here"

# Optional: ChromaDB settings
export CHROMADB_PERSIST_DIR="./chromadb_data"
export CHROMADB_COLLECTION="medical_xray_findings"

# Optional: Flower settings
export FLOWER_SERVER_ADDRESS="[::]:8080"
export FLOWER_NUM_ROUNDS="10"
```

### Configuration File

`config/langchain_rag_example.yaml`:

```yaml
rag:
  use_langchain: true
  chromadb:
    persist_directory: "./chromadb_data"
    top_k: 5
  gemini:
    api_key: ${GEMINI_API_KEY}
    model: "gemini-1.5-flash"
    temperature: 0.3
    max_tokens: 500

federated_learning:
  use_flower: true
  num_rounds: 10
  num_clients: 4
  differential_privacy:
    enabled: true
    noise_multiplier: 0.1
```

### Python Configuration

```python
from config.rag_config import RAGConfig

# Check configuration
is_valid, error = RAGConfig.validate()
if not is_valid:
    print(f"Config error: {error}")

# Get summary
summary = RAGConfig.get_summary()
print(summary)
```

## Troubleshooting

### ChromaDB Issues

**Problem:** "Collection already exists"
```python
# Solution: Use get_or_create
kb = ChromaDBMedicalKnowledgeBase()  # Automatically uses get_or_create
```

**Problem:** Slow queries
```bash
# Solution: Check collection size
stats = kb.get_stats()
print(f"Entries: {stats['total_entries']}")
# ChromaDB handles 10k+ entries efficiently
# If > 100k entries, consider sharding
```

### Gemini API Issues

**Problem:** "API key not set"
```bash
# Solution: Export key
export GEMINI_API_KEY="your-key"
```

**Problem:** Rate limit exceeded
```python
# Solution: Use fallback
try:
    result = pipeline.query(..., generate_explanation=True)
except Exception as e:
    # Falls back to simple RAG automatically
    result = pipeline.query(..., generate_explanation=False)
```

### Flower Issues

**Problem:** Clients not connecting
```bash
# Check firewall
sudo ufw allow 8080

# Use correct address
# Server: '[::]:8080'  (listen on all interfaces)
# Client: 'server_ip:8080'  (connect to specific IP)
```

**Problem:** Training hangs
```python
# Solution: Check min_clients setting
strategy = FlowerVFLStrategy(
    min_fit_clients=2,  # Reduce if testing with fewer clients
    min_available_clients=2
)
```

### General Issues

**Problem:** Import errors
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually
pip install chromadb google-generativeai langchain langchain-google-genai flwr
```

**Problem:** CUDA out of memory
```python
# Solution: Reduce batch size
client = FlowerMedicalClient(
    ...,
    learning_rate=0.001
)
# And in config:
config = {'batch_size': 16, 'local_epochs': 3}  # Reduced from 32
```

## Performance Benchmarks

**ChromaDB Retrieval:**
- 10 entries: < 1ms
- 1,000 entries: < 10ms  
- 10,000 entries: < 100ms
- 100,000 entries: < 500ms

**Gemini Generation:**
- Average latency: 1-3 seconds
- Depends on explanation length and API load

**Flower Federation:**
- 4 clients, 10 rounds: ~5-10 minutes (CPU)
- 4 clients, 10 rounds: ~2-5 minutes (GPU)

## Next Steps

1. **Explore Examples**: Try the usage examples above
2. **Customize**: Modify prompts in `GeminiMedicalExplainer`
3. **Scale**: Add more medical knowledge to ChromaDB
4. **Experiment**: Try different Flower strategies
5. **Integrate**: Connect to your own blockchain network

For questions or issues, see the main [README.md](../README.md) or open an issue on GitHub.
