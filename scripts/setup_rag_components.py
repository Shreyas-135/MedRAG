#!/usr/bin/env python3
"""
Setup Script for RAG Components

This script initializes the advanced RAG pipeline:
- Checks for Gemini API key
- Initializes ChromaDB with sample medical knowledge
- Populates vector database with COVID-19 findings
- Verifies all dependencies installed
- Runs validation tests
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    print("=" * 60)
    
    required_packages = {
        'chromadb': 'ChromaDB',
        'google.generativeai': 'Google Generative AI',
        'langchain': 'LangChain',
        'langchain_google_genai': 'LangChain Google GenAI',
        'flwr': 'Flower',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(package)
    
    if missing:
        print("\n⚠️  Missing packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def check_gemini_api_key():
    """Check if Gemini API key is set."""
    print("\nChecking Gemini API key...")
    print("=" * 60)
    
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("  ✗ GEMINI_API_KEY not set in environment")
        print("\nTo set your API key:")
        print("  1. Get your key from: https://makersuite.google.com/app/apikey")
        print("  2. Set it in your environment:")
        print("     export GEMINI_API_KEY='your-api-key-here'")
        print("  3. Or create a .env file with:")
        print("     GEMINI_API_KEY=your-api-key-here")
        return False
    
    if api_key == 'your-gemini-api-key-here':
        print("  ✗ GEMINI_API_KEY is set to placeholder value")
        print("\nPlease replace with your actual API key from:")
        print("  https://makersuite.google.com/app/apikey")
        return False
    
    print(f"  ✓ GEMINI_API_KEY is set ({api_key[:8]}...)")
    return True


def initialize_chromadb():
    """Initialize ChromaDB with sample medical knowledge."""
    print("\nInitializing ChromaDB...")
    print("=" * 60)
    
    try:
        from langchain_rag import (
            ChromaDBMedicalKnowledgeBase, 
            populate_sample_knowledge_base
        )
        
        # Create knowledge base
        kb = ChromaDBMedicalKnowledgeBase()
        
        # Check if already populated
        stats = kb.get_stats()
        if stats['total_entries'] > 0:
            print(f"  ℹ️  ChromaDB already populated with {stats['total_entries']} entries")
            print("  Skipping population (delete chromadb_data/ to repopulate)")
            return True
        
        # Populate with sample data
        print("  Populating with sample COVID-19 medical knowledge...")
        num_entries = populate_sample_knowledge_base(kb)
        
        # Verify
        stats = kb.get_stats()
        print(f"\n  ✓ ChromaDB initialized successfully")
        print(f"  Collection: {stats['collection_name']}")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Persist directory: {stats['persist_directory']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to initialize ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_langchain_pipeline():
    """Test LangChain RAG pipeline."""
    print("\nTesting LangChain RAG Pipeline...")
    print("=" * 60)
    
    try:
        from langchain_rag import LangChainRAGPipeline, ChromaDBMedicalKnowledgeBase
        import numpy as np
        
        # Initialize components
        kb = ChromaDBMedicalKnowledgeBase()
        
        # Test without Gemini (in case API key not set)
        print("  Testing retrieval (without LLM)...")
        pipeline = LangChainRAGPipeline(knowledge_base=kb, explainer=None)
        
        # Test query
        query_emb = np.random.randn(64)
        result = pipeline.query(
            embedding=query_emb,
            prediction='COVID-19',
            confidence=0.87,
            top_k=3,
            generate_explanation=False
        )
        
        print(f"  ✓ Pipeline query successful")
        print(f"    Retrieved {len(result['retrieved_cases'])} cases")
        print(f"    Explanation hash: {result['explanation_hash'][:16]}...")
        
        # Test with Gemini if API key is available
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key and gemini_key != 'your-gemini-api-key-here':
            print("\n  Testing with Gemini LLM...")
            try:
                from langchain_rag import GeminiMedicalExplainer
                explainer = GeminiMedicalExplainer()
                pipeline_with_llm = LangChainRAGPipeline(knowledge_base=kb, explainer=explainer)
                
                result = pipeline_with_llm.query(
                    embedding=query_emb,
                    prediction='COVID-19',
                    confidence=0.87,
                    top_k=3,
                    generate_explanation=True
                )
                
                print(f"  ✓ LLM explanation generated")
                print(f"    Explanation length: {len(result['rag_explanation'])} chars")
                print(f"    Citations: {len(result['citations'])}")
            except Exception as e:
                print(f"  ⚠️  LLM test skipped: {e}")
        else:
            print("  ⊗ Skipping LLM test (API key not set)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flower_components():
    """Test Flower federated learning components."""
    print("\nTesting Flower Components...")
    print("=" * 60)
    
    try:
        from flower_vfl import FlowerMedicalClient, FlowerVFLStrategy
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 64)
            
            def forward(self, x):
                return self.fc(x)
        
        model = TestModel()
        
        # Create dummy data
        dummy_data = TensorDataset(
            torch.randn(50, 10),
            torch.randint(0, 2, (50,))
        )
        train_loader = DataLoader(dummy_data, batch_size=10)
        
        # Test client creation
        print("  Testing FlowerMedicalClient...")
        client = FlowerMedicalClient(
            client_id='Test_Hospital',
            model=model,
            train_loader=train_loader,
            apply_dp=True
        )
        print(f"    ✓ Client created: {client.client_id}")
        
        # Test strategy
        print("  Testing FlowerVFLStrategy...")
        strategy = FlowerVFLStrategy(
            min_fit_clients=2,
            min_available_clients=2
        )
        print("    ✓ Strategy created")
        
        print("\n  ✓ Flower components working")
        return True
        
    except Exception as e:
        print(f"  ✗ Flower test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print setup summary."""
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    status_map = {
        'dependencies': 'Dependencies',
        'gemini_key': 'Gemini API Key',
        'chromadb': 'ChromaDB',
        'langchain': 'LangChain Pipeline',
        'flower': 'Flower Components'
    }
    
    for key, name in status_map.items():
        status = "✓" if results.get(key, False) else "✗"
        print(f"  {status} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All components initialized successfully!")
        print("\nYou can now:")
        print("  1. Run training with LangChain RAG:")
        print("     python src/demo_flower_vfl.py --use-langchain")
        print("  2. Run inference with LLM explanations:")
        print("     python src/inference.py --use-langchain")
        print("  3. Start the web UI:")
        print("     streamlit run webapp/app.py")
    else:
        print("\n⚠️  Some components failed to initialize")
        print("Please fix the issues above and run this script again.")
        
        if not results.get('gemini_key'):
            print("\nNote: Gemini API key is optional. You can still use:")
            print("  - Simple RAG without LLM explanations")
            print("  - ChromaDB for retrieval")
            print("  - Flower for federated learning")
    
    return all_passed


def main():
    """Main setup function."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "MedRAG Advanced Components Setup" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    results = {}
    
    # Step 1: Check dependencies
    results['dependencies'] = check_dependencies()
    if not results['dependencies']:
        print("\n⚠️  Cannot proceed without required dependencies")
        print("Please install them and run this script again.")
        return 1
    
    # Step 2: Check Gemini API key (optional)
    results['gemini_key'] = check_gemini_api_key()
    
    # Step 3: Initialize ChromaDB
    results['chromadb'] = initialize_chromadb()
    
    # Step 4: Test LangChain pipeline
    results['langchain'] = test_langchain_pipeline()
    
    # Step 5: Test Flower components
    results['flower'] = test_flower_components()
    
    # Print summary
    success = print_summary(results)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
