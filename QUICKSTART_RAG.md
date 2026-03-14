# Quick Start: Advanced RAG Features

This guide helps you quickly get started with the new LangChain RAG and Flower federated learning features.

## Prerequisites

1. **Python 3.8+** installed
2. **Git** repository cloned
3. **Internet connection** for downloading dependencies

## Step 1: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

This installs:
- LangChain for RAG orchestration
- ChromaDB for vector storage
- Google Generative AI for Gemini
- Flower for federated learning
- All existing dependencies

## Step 2: Get Gemini API Key (Optional but Recommended)

1. Visit https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

## Step 3: Set Environment Variables

```bash
# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:GEMINI_API_KEY="your-api-key-here"

# Or create .env file in project root
echo "GEMINI_API_KEY=your-api-key-here" > .env
```

## Step 4: Initialize ChromaDB

```bash
# Run the setup script
python scripts/setup_rag_components.py
```

This will:
- ✓ Check all dependencies
- ✓ Validate Gemini API key
- ✓ Create ChromaDB database
- ✓ Populate with sample medical knowledge
- ✓ Test all components

Expected output:
```
╔══════════════════════════════════════════════════════╗
║          MedRAG Advanced Components Setup            ║
╚══════════════════════════════════════════════════════╝

Checking dependencies...
  ✓ ChromaDB
  ✓ Google Generative AI
  ✓ LangChain
  ✓ LangChain Google GenAI
  ✓ Flower

✓ All dependencies installed

Checking Gemini API key...
  ✓ GEMINI_API_KEY is set (sk_xxxx...)

Initializing ChromaDB...
  ✓ ChromaDB initialized successfully
  Collection: medical_xray_findings
  Total entries: 10

Testing LangChain RAG Pipeline...
  ✓ Pipeline query successful

Testing Flower Components...
  ✓ Flower components working

✅ All components initialized successfully!
```

## Step 5: Try It Out!

### Option A: Inference with LLM Explanations

```python
# test_rag.py
from inference import load_inference_model
import numpy as np

# Load model with LangChain RAG
print("Loading model with LangChain RAG...")
engine = load_inference_model(
    use_rag=True, 
    use_langchain=True,
    num_clients=4
)

# Note: This is a demo - replace with actual X-ray image path
# For now, we'll test the pipeline structure

# Get RAG pipeline stats
if hasattr(engine.server_model, 'get_chromadb_stats'):
    stats = engine.server_model.get_chromadb_stats()
    print(f"ChromaDB Stats: {stats}")

print("✓ Model loaded successfully with LangChain RAG!")
```

Run it:
```bash
python test_rag.py
```

### Option B: Test ChromaDB Directly

```python
# test_chromadb.py
from langchain_rag import ChromaDBMedicalKnowledgeBase
import numpy as np

# Initialize knowledge base
kb = ChromaDBMedicalKnowledgeBase()

# Check what's in there
stats = kb.get_stats()
print(f"Total entries: {stats['total_entries']}")

# Search for COVID-19 cases
query = np.random.randn(64)  # In practice, this is image embedding
results = kb.search(query, top_k=3)

for i, result in enumerate(results, 1):
    print(f"\nCase {i}:")
    print(f"  Similarity: {result['similarity']:.2f}")
    print(f"  Condition: {result['metadata']['condition']}")
    print(f"  Severity: {result['metadata']['severity']}")
    print(f"  Text: {result['text'][:80]}...")
```

Run it:
```bash
python test_chromadb.py
```

### Option C: Test Gemini Explanation (Requires API Key)

```python
# test_gemini.py
from langchain_rag import GeminiMedicalExplainer

try:
    explainer = GeminiMedicalExplainer()
    
    # Test explanation generation
    retrieved_cases = [
        {
            'text': 'Bilateral ground-glass opacities consistent with COVID-19',
            'similarity': 0.95,
            'metadata': {'condition': 'covid', 'severity': 'moderate'}
        }
    ]
    
    result = explainer.generate_explanation(
        prediction='COVID-19',
        confidence=0.87,
        retrieved_cases=retrieved_cases
    )
    
    print("Generated Explanation:")
    print(result['explanation'])
    print(f"\nCitations: {len(result['citations'])} cases")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure GEMINI_API_KEY is set correctly")
```

Run it:
```bash
python test_gemini.py
```

## Common Issues

### Issue: "ChromaDB not available"

**Solution:**
```bash
pip install chromadb
```

### Issue: "GEMINI_API_KEY not set"

**Solution:**
```bash
# Set it temporarily
export GEMINI_API_KEY="your-key"

# Or add to .env file
echo "GEMINI_API_KEY=your-key" > .env
```

### Issue: "Flower not available"

**Solution:**
```bash
pip install flwr flwr-datasets
```

### Issue: ChromaDB permission errors

**Solution:**
```bash
# Make sure chromadb_data directory is writable
chmod -R 755 ./chromadb_data
```

## Next Steps

1. **Read the Full Guide**: See [docs/LANGCHAIN_RAG_GUIDE.md](docs/LANGCHAIN_RAG_GUIDE.md)
2. **Try Training**: Run federated training with Flower (once demo script is ready)
3. **Customize**: Modify prompts in `GeminiMedicalExplainer`
4. **Add Knowledge**: Populate ChromaDB with your own medical data
5. **Web UI**: Check the Streamlit app for visual interface

## Configuration Options

All settings are in `config/langchain_rag_example.yaml`:

```yaml
rag:
  use_langchain: true
  chromadb:
    persist_directory: "./chromadb_data"
    collection_name: "medical_xray_findings"
    top_k: 5
  gemini:
    api_key: ${GEMINI_API_KEY}
    model: "gemini-1.5-flash"
    temperature: 0.3
    max_tokens: 500
```

Or configure via Python:

```python
from config.rag_config import RAGConfig

# Check current config
summary = RAGConfig.get_summary()
print(summary)

# Validate
is_valid, error = RAGConfig.validate()
if not is_valid:
    print(f"Config error: {error}")
```

## Help & Support

- **Documentation**: [docs/LANGCHAIN_RAG_GUIDE.md](docs/LANGCHAIN_RAG_GUIDE.md)
- **Implementation Details**: [IMPLEMENTATION_SUMMARY_RAG.md](IMPLEMENTATION_SUMMARY_RAG.md)
- **Main README**: [README.md](README.md)
- **Issues**: GitHub Issues page

## What You Can Do Now

✅ **Use ChromaDB** - Persistent vector database for medical knowledge  
✅ **Get LLM Explanations** - Gemini generates clinical explanations  
✅ **Run Tests** - Comprehensive test suite with mocking  
✅ **Configure Everything** - Via YAML or environment variables  
✅ **Maintain Compatibility** - All old features still work  

Enjoy the enhanced RAG capabilities! 🚀
