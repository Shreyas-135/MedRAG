"""
Tests for LangChain RAG Components

This module tests:
- ChromaDB initialization and storage
- Gemini API integration (mocked)
- LangChain pipeline end-to-end
- Blockchain hash integration
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestChromaDBKnowledgeBase(unittest.TestCase):
    """Test ChromaDB medical knowledge base."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a test-specific directory
        self.test_dir = './test_chromadb_data'
        
    def tearDown(self):
        """Clean up test data."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_chromadb_initialization(self):
        """Test ChromaDB initialization."""
        try:
            from langchain_rag import ChromaDBMedicalKnowledgeBase
            
            kb = ChromaDBMedicalKnowledgeBase(
                persist_directory=self.test_dir,
                collection_name='test_collection'
            )
            
            stats = kb.get_stats()
            self.assertEqual(stats['collection_name'], 'test_collection')
            self.assertEqual(stats['total_entries'], 0)
            
            print("  ✓ ChromaDB initialization test passed")
            
        except ImportError:
            self.skipTest("ChromaDB not available")
    
    def test_add_entry(self):
        """Test adding entries to knowledge base."""
        try:
            from langchain_rag import ChromaDBMedicalKnowledgeBase
            
            kb = ChromaDBMedicalKnowledgeBase(
                persist_directory=self.test_dir,
                collection_name='test_collection'
            )
            
            # Add entry
            text = "Test medical finding"
            embedding = np.random.randn(64)
            metadata = {'condition': 'test', 'severity': 'low'}
            
            entry_id = kb.add_entry(text, embedding, metadata)
            
            self.assertIsNotNone(entry_id)
            
            stats = kb.get_stats()
            self.assertEqual(stats['total_entries'], 1)
            
            print("  ✓ Add entry test passed")
            
        except ImportError:
            self.skipTest("ChromaDB not available")
    
    def test_search(self):
        """Test similarity search."""
        try:
            from langchain_rag import ChromaDBMedicalKnowledgeBase, populate_sample_knowledge_base
            
            kb = ChromaDBMedicalKnowledgeBase(
                persist_directory=self.test_dir,
                collection_name='test_collection'
            )
            
            # Populate with sample data
            populate_sample_knowledge_base(kb)
            
            # Search
            query_emb = np.random.randn(64)
            results = kb.search(query_emb, top_k=3)
            
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 3)
            
            if len(results) > 0:
                self.assertIn('text', results[0])
                self.assertIn('similarity', results[0])
                self.assertIn('metadata', results[0])
            
            print("  ✓ Search test passed")
            
        except ImportError:
            self.skipTest("ChromaDB not available")
    
    def test_get_hash(self):
        """Test knowledge base hash generation."""
        try:
            from langchain_rag import ChromaDBMedicalKnowledgeBase
            
            kb = ChromaDBMedicalKnowledgeBase(
                persist_directory=self.test_dir,
                collection_name='test_collection'
            )
            
            # Add entry
            kb.add_entry("Test", np.random.randn(64), {'condition': 'test'})
            
            # Get hash
            hash1 = kb.get_hash()
            self.assertIsInstance(hash1, str)
            self.assertEqual(len(hash1), 64)  # SHA-256 hash
            
            # Hash should be deterministic
            hash2 = kb.get_hash()
            self.assertEqual(hash1, hash2)
            
            print("  ✓ Hash generation test passed")
            
        except ImportError:
            self.skipTest("ChromaDB not available")


class TestGeminiExplainer(unittest.TestCase):
    """Test Gemini medical explainer with mocking."""
    
    @patch('langchain_rag.genai')
    def test_gemini_initialization_mocked(self, mock_genai):
        """Test Gemini initialization with mocked API."""
        try:
            from langchain_rag import GeminiMedicalExplainer
            
            # Mock the model
            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            
            # Set fake API key for testing
            os.environ['GEMINI_API_KEY'] = 'test-api-key-for-testing'
            
            explainer = GeminiMedicalExplainer()
            
            self.assertIsNotNone(explainer)
            self.assertEqual(explainer.model_name, 'gemini-1.5-flash')
            
            print("  ✓ Gemini initialization test passed")
            
        except ImportError:
            self.skipTest("Gemini dependencies not available")
        finally:
            # Clean up
            if 'GEMINI_API_KEY' in os.environ:
                del os.environ['GEMINI_API_KEY']
    
    @patch('langchain_rag.genai')
    def test_generate_explanation_mocked(self, mock_genai):
        """Test explanation generation with mocked API."""
        try:
            from langchain_rag import GeminiMedicalExplainer
            
            # Mock the model and response
            mock_response = Mock()
            mock_response.text = "This is a test medical explanation."
            
            mock_model = Mock()
            mock_model.generate_content.return_value = mock_response
            
            mock_genai.GenerativeModel.return_value = mock_model
            
            # Set fake API key
            os.environ['GEMINI_API_KEY'] = 'test-api-key-for-testing'
            
            explainer = GeminiMedicalExplainer()
            
            # Test explanation generation
            retrieved_cases = [
                {
                    'text': 'Test case',
                    'similarity': 0.9,
                    'metadata': {'condition': 'covid', 'severity': 'moderate'}
                }
            ]
            
            result = explainer.generate_explanation(
                prediction='COVID-19',
                confidence=0.85,
                retrieved_cases=retrieved_cases
            )
            
            self.assertIn('explanation', result)
            self.assertIn('citations', result)
            self.assertIn('confidence', result)
            
            print("  ✓ Generate explanation test passed")
            
        except ImportError:
            self.skipTest("Gemini dependencies not available")
        finally:
            # Clean up
            if 'GEMINI_API_KEY' in os.environ:
                del os.environ['GEMINI_API_KEY']


class TestLangChainPipeline(unittest.TestCase):
    """Test LangChain RAG pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = './test_chromadb_data'
    
    def tearDown(self):
        """Clean up test data."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_pipeline_without_llm(self):
        """Test pipeline with retrieval only (no LLM)."""
        try:
            from langchain_rag import (
                LangChainRAGPipeline,
                ChromaDBMedicalKnowledgeBase,
                populate_sample_knowledge_base
            )
            
            # Create knowledge base
            kb = ChromaDBMedicalKnowledgeBase(
                persist_directory=self.test_dir,
                collection_name='test_collection'
            )
            populate_sample_knowledge_base(kb)
            
            # Create pipeline without explainer
            pipeline = LangChainRAGPipeline(knowledge_base=kb, explainer=None)
            
            # Test query
            query_emb = np.random.randn(64)
            result = pipeline.query(
                embedding=query_emb,
                prediction='COVID-19',
                confidence=0.87,
                generate_explanation=False
            )
            
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)
            self.assertIn('retrieved_cases', result)
            self.assertIn('explanation_hash', result)
            
            print("  ✓ Pipeline (no LLM) test passed")
            
        except ImportError:
            self.skipTest("LangChain RAG dependencies not available")
    
    @patch('langchain_rag.genai')
    def test_pipeline_with_llm_mocked(self, mock_genai):
        """Test pipeline with mocked LLM."""
        try:
            from langchain_rag import (
                LangChainRAGPipeline,
                ChromaDBMedicalKnowledgeBase,
                GeminiMedicalExplainer,
                populate_sample_knowledge_base
            )
            
            # Mock Gemini
            mock_response = Mock()
            mock_response.text = "Test explanation from Gemini."
            mock_model = Mock()
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            
            # Set fake API key
            os.environ['GEMINI_API_KEY'] = 'test-api-key-for-testing'
            
            # Create components
            kb = ChromaDBMedicalKnowledgeBase(
                persist_directory=self.test_dir,
                collection_name='test_collection'
            )
            populate_sample_knowledge_base(kb)
            
            explainer = GeminiMedicalExplainer()
            pipeline = LangChainRAGPipeline(knowledge_base=kb, explainer=explainer)
            
            # Test query with LLM
            query_emb = np.random.randn(64)
            result = pipeline.query(
                embedding=query_emb,
                prediction='COVID-19',
                confidence=0.87,
                generate_explanation=True
            )
            
            self.assertIn('rag_explanation', result)
            self.assertIn('citations', result)
            
            print("  ✓ Pipeline (with LLM) test passed")
            
        except ImportError:
            self.skipTest("LangChain RAG dependencies not available")
        finally:
            if 'GEMINI_API_KEY' in os.environ:
                del os.environ['GEMINI_API_KEY']
    
    def test_blockchain_hash_integration(self):
        """Test blockchain hash generation."""
        try:
            from langchain_rag import (
                LangChainRAGPipeline,
                ChromaDBMedicalKnowledgeBase,
                populate_sample_knowledge_base
            )
            
            kb = ChromaDBMedicalKnowledgeBase(
                persist_directory=self.test_dir,
                collection_name='test_collection'
            )
            populate_sample_knowledge_base(kb)
            
            pipeline = LangChainRAGPipeline(knowledge_base=kb)
            
            # Test query
            query_emb = np.random.randn(64)
            result = pipeline.query(
                embedding=query_emb,
                prediction='COVID-19',
                confidence=0.87,
                generate_explanation=False
            )
            
            # Check hashes
            self.assertIn('explanation_hash', result)
            self.assertIn('knowledge_base_hash', result)
            
            self.assertIsInstance(result['explanation_hash'], str)
            self.assertIsInstance(result['knowledge_base_hash'], str)
            
            # Hashes should be SHA-256 (64 hex chars)
            self.assertEqual(len(result['explanation_hash']), 64)
            self.assertEqual(len(result['knowledge_base_hash']), 64)
            
            print("  ✓ Blockchain hash integration test passed")
            
        except ImportError:
            self.skipTest("LangChain RAG dependencies not available")


def run_tests():
    """Run all tests."""
    print("Running LangChain RAG Tests")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestChromaDBKnowledgeBase))
    suite.addTests(loader.loadTestsFromTestCase(TestGeminiExplainer))
    suite.addTests(loader.loadTestsFromTestCase(TestLangChainPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
