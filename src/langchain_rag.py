"""
LangChain RAG Implementation for Medical Image Analysis

This module provides LangChain-based Retrieval-Augmented Generation
with ChromaDB vector storage and Gemini LLM for medical explanations.
"""

import os
import sys
import hashlib
import json
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not available. Install with: pip install chromadb")

try:
    import google.generativeai as genai
    from langchain_google_genai import GoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Gemini not available. Install with: pip install google-generativeai langchain-google-genai")

try:
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Install with: pip install langchain")

from config.rag_config import RAGConfig

try:
    from provenance import (
        hash_retrieval_params,
        hash_prompt,
        hash_generation_params,
        hash_model_version,
        build_provenance_bundle,
    )
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False


class ChromaDBMedicalKnowledgeBase:
    """
    ChromaDB-based medical knowledge base for X-ray findings.
    
    Stores embeddings with metadata (condition, severity, findings)
    and provides similarity search for RAG retrieval.
    """
    
    def __init__(self, 
                 persist_directory: str = None,
                 collection_name: str = None):
        """
        Initialize ChromaDB knowledge base.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.persist_directory = persist_directory or RAGConfig.CHROMADB_PERSIST_DIR
        self.collection_name = collection_name or RAGConfig.CHROMADB_COLLECTION
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Medical X-ray findings knowledge base"}
        )
        
        print(f"✓ ChromaDB initialized: {self.persist_directory}/{self.collection_name}")
        print(f"  Current entries: {self.collection.count()}")
    
    def add_entry(self, 
                  text: str, 
                  embedding: np.ndarray,
                  metadata: Dict[str, Any],
                  entry_id: str = None) -> str:
        """
        Add a medical knowledge entry.
        
        Args:
            text: Medical text description
            embedding: Feature embedding vector
            metadata: Metadata (condition, severity, findings)
            entry_id: Optional unique ID (auto-generated if None)
            
        Returns:
            Entry ID
        """
        # Generate ID if not provided
        if entry_id is None:
            entry_id = hashlib.md5(text.encode()).hexdigest()
        
        # Convert numpy array to list for ChromaDB
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        # Add to collection
        self.collection.add(
            embeddings=[embedding_list],
            documents=[text],
            metadatas=[metadata],
            ids=[entry_id]
        )
        
        return entry_id
    
    def add_batch(self, 
                  texts: List[str],
                  embeddings: List[np.ndarray],
                  metadatas: List[Dict[str, Any]],
                  ids: List[str] = None) -> List[str]:
        """
        Add multiple entries in batch.
        
        Args:
            texts: List of medical descriptions
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs
            
        Returns:
            List of entry IDs
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [hashlib.md5(text.encode()).hexdigest() for text in texts]
        
        # Convert embeddings to lists
        embedding_lists = [
            emb.tolist() if isinstance(emb, np.ndarray) else emb 
            for emb in embeddings
        ]
        
        # Add to collection
        self.collection.add(
            embeddings=embedding_lists,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def search(self, 
               query_embedding: np.ndarray,
               top_k: int = None,
               filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar medical knowledge entries.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of result dictionaries with text, similarity, metadata
        """
        top_k = top_k or RAGConfig.CHROMADB_TOP_K
        
        # Convert to list for ChromaDB
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'similarity': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i],
                    'id': results['ids'][0][i]
                })
        
        return formatted_results
    
    def update_entry(self, entry_id: str, text: str = None, 
                    embedding: np.ndarray = None,
                    metadata: Dict[str, Any] = None):
        """
        Update an existing entry.
        
        Args:
            entry_id: ID of entry to update
            text: New text (optional)
            embedding: New embedding (optional)
            metadata: New metadata (optional)
        """
        update_params = {'ids': [entry_id]}
        
        if text is not None:
            update_params['documents'] = [text]
        if embedding is not None:
            update_params['embeddings'] = [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding]
        if metadata is not None:
            update_params['metadatas'] = [metadata]
        
        self.collection.update(**update_params)
    
    def delete_entry(self, entry_id: str):
        """Delete an entry by ID."""
        self.collection.delete(ids=[entry_id])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            'total_entries': self.collection.count(),
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory
        }
    
    def get_hash(self) -> str:
        """Generate hash of knowledge base for blockchain verification."""
        # Get all entries
        all_data = self.collection.get()
        
        # Create deterministic hash
        content = json.dumps({
            'documents': sorted(all_data['documents']) if all_data['documents'] else [],
            'ids': sorted(all_data['ids']) if all_data['ids'] else []
        }, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()


class GeminiMedicalExplainer:
    """
    Gemini-based medical explanation generator.
    
    Uses Google Gemini API to generate natural language explanations
    from retrieved medical knowledge.
    """
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize Gemini explainer.
        
        Args:
            api_key: Gemini API key (uses env var if None)
            model_name: Model name (uses config default if None)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Gemini not available. Install with: pip install google-generativeai")
        
        self.api_key = api_key or RAGConfig.GEMINI_API_KEY
        if not self.api_key or self.api_key == 'your-gemini-api-key-here':
            raise ValueError("GEMINI_API_KEY not set. Please set it in environment or pass as parameter.")
        
        self.model_name = model_name or RAGConfig.GEMINI_MODEL
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(self.model_name)
        
        # Generation config
        self.generation_config = {
            'temperature': RAGConfig.GEMINI_TEMPERATURE,
            'max_output_tokens': RAGConfig.GEMINI_MAX_TOKENS,
            'top_p': RAGConfig.GEMINI_TOP_P,
            'top_k': RAGConfig.GEMINI_TOP_K,
        }
        
        print(f"✓ Gemini initialized: {self.model_name}")
    
    def generate_explanation(self,
                           prediction: str,
                           confidence: float,
                           retrieved_cases: List[Dict[str, Any]],
                           include_citations: bool = True) -> Dict[str, Any]:
        """
        Generate medical explanation from prediction and retrieved cases.
        
        Args:
            prediction: Predicted class/condition
            confidence: Prediction confidence (0-1)
            retrieved_cases: List of retrieved medical cases
            include_citations: Whether to include citations
            
        Returns:
            Dictionary with explanation, citations, and metadata
        """
        # Build context from retrieved cases
        context = self._build_context(retrieved_cases)
        
        # Create prompt
        prompt = self._create_explanation_prompt(
            prediction, confidence, context, include_citations
        )
        
        try:
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            explanation_text = response.text
            
            # Extract citations if requested
            citations = []
            if include_citations:
                citations = self._extract_citations(retrieved_cases)
            
            return {
                'explanation': explanation_text,
                'citations': citations,
                'confidence': confidence,
                'prediction': prediction,
                'num_retrieved_cases': len(retrieved_cases),
                'model': self.model_name
            }
        
        except Exception as e:
            print(f"Warning: Gemini API error: {e}")
            # Fallback to simple explanation
            return self._fallback_explanation(prediction, confidence, retrieved_cases)
    
    def _build_context(self, retrieved_cases: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved cases."""
        if not retrieved_cases:
            return "No similar cases found in knowledge base."
        
        context_parts = []
        for i, case in enumerate(retrieved_cases, 1):
            similarity = case.get('similarity', 0) * 100
            text = case.get('text', 'No description')
            metadata = case.get('metadata', {})
            
            context_parts.append(
                f"Case {i} (similarity: {similarity:.1f}%):\n"
                f"  Condition: {metadata.get('condition', 'Unknown')}\n"
                f"  Severity: {metadata.get('severity', 'Unknown')}\n"
                f"  Description: {text}"
            )
        
        return "\n\n".join(context_parts)
    
    def _create_explanation_prompt(self,
                                  prediction: str,
                                  confidence: float,
                                  context: str,
                                  include_citations: bool) -> str:
        """Create prompt for Gemini."""
        prompt = f"""You are a medical AI assistant analyzing chest X-ray images. 
Based on the model prediction and similar cases from the medical knowledge base, provide a clear, clinically-relevant explanation.

**Model Prediction**: {prediction}
**Confidence**: {confidence * 100:.1f}%

**Similar Cases from Medical Knowledge Base**:
{context}

Please provide:
1. A brief clinical interpretation of the prediction
2. Key radiological findings that support this diagnosis
3. Clinical significance and recommendations
"""

        if include_citations:
            prompt += "\n4. References to the similar cases used in this analysis"
        
        prompt += "\n\nKeep the explanation concise, professional, and suitable for clinical use. Limit to 3-4 paragraphs."
        
        return prompt
    
    def _extract_citations(self, retrieved_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citation information from retrieved cases."""
        citations = []
        for i, case in enumerate(retrieved_cases, 1):
            citations.append({
                'case_number': i,
                'similarity': case.get('similarity', 0),
                'condition': case.get('metadata', {}).get('condition', 'Unknown'),
                'text_snippet': case.get('text', '')[:100] + '...'
            })
        return citations
    
    def _fallback_explanation(self,
                            prediction: str,
                            confidence: float,
                            retrieved_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate fallback explanation when Gemini is unavailable."""
        explanation = f"""Based on the analysis, the model predicts: {prediction} with {confidence * 100:.1f}% confidence.

This prediction is informed by {len(retrieved_cases)} similar cases in the medical knowledge base."""
        
        if retrieved_cases:
            top_case = retrieved_cases[0]
            explanation += f"\n\nThe most similar case shows: {top_case.get('text', 'No description')}"
        
        return {
            'explanation': explanation,
            'citations': self._extract_citations(retrieved_cases),
            'confidence': confidence,
            'prediction': prediction,
            'num_retrieved_cases': len(retrieved_cases),
            'model': 'fallback'
        }


class LangChainRAGPipeline:
    """
    Complete LangChain RAG pipeline orchestrating retrieval and generation.
    
    Workflow: Query → ChromaDB Retrieval → Gemini Explanation → Formatted Response
    """
    
    def __init__(self,
                 knowledge_base: ChromaDBMedicalKnowledgeBase = None,
                 explainer: GeminiMedicalExplainer = None,
                 blockchain_integrator = None):
        """
        Initialize LangChain RAG pipeline.
        
        Args:
            knowledge_base: ChromaDB knowledge base
            explainer: Gemini explainer
            blockchain_integrator: Optional blockchain integrator
        """
        self.knowledge_base = knowledge_base or ChromaDBMedicalKnowledgeBase()
        self.explainer = explainer
        self.blockchain_integrator = blockchain_integrator
        
        # Try to initialize explainer if not provided and Gemini is available
        if self.explainer is None and GEMINI_AVAILABLE:
            try:
                self.explainer = GeminiMedicalExplainer()
            except (ValueError, ImportError) as e:
                print(f"Warning: Could not initialize Gemini: {e}")
                print("Pipeline will use simple retrieval without LLM explanations.")
        
        print("✓ LangChain RAG Pipeline initialized")
    
    def query(self,
             embedding: np.ndarray,
             prediction: str,
             confidence: float,
             top_k: int = None,
             generate_explanation: bool = True,
             provenance_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute full RAG pipeline query.
        
        Args:
            embedding: Query embedding from X-ray image
            prediction: Model prediction
            confidence: Prediction confidence
            top_k: Number of cases to retrieve
            generate_explanation: Whether to generate LLM explanation
            provenance_config: Optional dict with provenance metadata:
                hospital_id, site_id, device_id, model_version_id,
                model_hash (all default to 'unknown').
            
        Returns:
            Complete RAG result with explanation, citations, and provenance hashes.
            Includes: retrieval_hash, prompt_hash, generation_params_hash,
            provenance_bundle.
        """
        top_k = top_k or RAGConfig.RAG_RETRIEVAL_TOP_K
        
        # Step 1: Retrieve similar cases from ChromaDB
        retrieved_cases = self.knowledge_base.search(
            query_embedding=embedding,
            top_k=top_k
        )
        
        # Step 2: Generate explanation with Gemini (if available and requested)
        if generate_explanation and self.explainer is not None:
            explanation_result = self.explainer.generate_explanation(
                prediction=prediction,
                confidence=confidence,
                retrieved_cases=retrieved_cases,
                include_citations=True
            )
        else:
            # Simple explanation without LLM
            explanation_result = {
                'explanation': f"Prediction: {prediction} ({confidence * 100:.1f}% confidence)",
                'citations': [],
                'confidence': confidence,
                'prediction': prediction,
                'num_retrieved_cases': len(retrieved_cases),
                'model': 'simple'
            }
        
        # Step 3: Add blockchain hash for audit trail
        explanation_hash = self._compute_explanation_hash(explanation_result)
        knowledge_base_hash = self.knowledge_base.get_hash()
        
        if self.blockchain_integrator:
            # Log to blockchain
            self._log_to_blockchain(explanation_hash, retrieved_cases)
        
        # Step 4: Compute provenance hashes
        provenance_config = provenance_config or {}
        retrieval_hash = None
        prompt_hash = None
        generation_params_hash = None
        provenance_bundle = None

        if PROVENANCE_AVAILABLE:
            # Retrieval hash: IDs + similarity scores + top_k
            item_ids = [c.get('id', '') for c in retrieved_cases]
            similarity_scores = [c.get('similarity', 0.0) for c in retrieved_cases]
            retrieval_hash = hash_retrieval_params(
                item_ids=item_ids,
                similarity_scores=similarity_scores,
                top_k=top_k,
            )

            # Prompt hash: hash the inputs that make up the prompt (prediction +
            # confidence), which is what was sent to the LLM. The retrieved cases
            # are already captured via retrieval_hash.
            prompt_hash = hash_prompt(
                f"{prediction}:{confidence:.6f}"
            )

            # Generation params hash
            model_id = explanation_result.get('model', 'simple')
            explainer_obj = self.explainer
            temperature = getattr(explainer_obj, 'generation_config', {}).get(
                'temperature', RAGConfig.GEMINI_TEMPERATURE
            ) if explainer_obj else RAGConfig.GEMINI_TEMPERATURE
            max_tokens = getattr(explainer_obj, 'generation_config', {}).get(
                'max_output_tokens', RAGConfig.GEMINI_MAX_TOKENS
            ) if explainer_obj else RAGConfig.GEMINI_MAX_TOKENS
            generation_params_hash = hash_generation_params(
                temperature=temperature,
                max_tokens=max_tokens,
                model_id=model_id,
            )

            # Model version hash
            model_version_hash = hash_model_version(
                version_id=provenance_config.get('model_version_id', 'unknown'),
                model_hash=provenance_config.get('model_hash', ''),
            )

            # Build provenance bundle
            provenance_bundle = build_provenance_bundle(
                knowledge_base_hash=knowledge_base_hash,
                explanation_hash=explanation_hash,
                retrieval_hash=retrieval_hash,
                model_version_hash=model_version_hash,
                prompt_hash=prompt_hash,
                generation_params_hash=generation_params_hash,
                hospital_id=provenance_config.get('hospital_id', 'unknown'),
                site_id=provenance_config.get('site_id', 'unknown'),
                device_id=provenance_config.get('device_id', 'unknown'),
            )

        # Compile final result
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'rag_explanation': explanation_result['explanation'],
            'retrieved_cases': retrieved_cases,
            'citations': explanation_result['citations'],
            'explanation_hash': explanation_hash,
            'knowledge_base_hash': knowledge_base_hash,
            'model_used': explanation_result['model'],
            # Provenance fields (None when provenance module is unavailable)
            'retrieval_hash': retrieval_hash,
            'prompt_hash': prompt_hash,
            'generation_params_hash': generation_params_hash,
            'provenance_bundle': provenance_bundle,
        }
        
        return result
    
    def _compute_explanation_hash(self, explanation_result: Dict[str, Any]) -> str:
        """Compute hash of explanation for blockchain verification."""
        content = json.dumps({
            'explanation': explanation_result['explanation'],
            'prediction': explanation_result['prediction'],
            'confidence': explanation_result['confidence']
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _log_to_blockchain(self, explanation_hash: str, retrieved_cases: List[Dict[str, Any]]):
        """Log RAG operation to blockchain."""
        if self.blockchain_integrator:
            # In practice, this would call blockchain smart contract
            print(f"Logging to blockchain: {explanation_hash[:16]}...")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        kb_stats = self.knowledge_base.get_stats()
        
        return {
            'knowledge_base': kb_stats,
            'explainer_available': self.explainer is not None,
            'explainer_model': self.explainer.model_name if self.explainer else None,
            'blockchain_enabled': self.blockchain_integrator is not None
        }


def populate_sample_knowledge_base(kb: ChromaDBMedicalKnowledgeBase):
    """
    Populate knowledge base with sample COVID-19 medical findings.
    
    Args:
        kb: ChromaDB knowledge base to populate
    """
    sample_entries = [
        {
            'text': 'Normal chest X-ray showing clear lung fields with no signs of consolidation, ground-glass opacities, or pleural effusion. Cardiac silhouette normal size.',
            'embedding': np.random.randn(64) * 0.3,
            'metadata': {
                'condition': 'normal',
                'severity': 'none',
                'findings': ['clear_lungs', 'normal_cardiac_silhouette'],
                'location': 'bilateral'
            }
        },
        {
            'text': 'Bilateral ground-glass opacities predominantly in the peripheral and lower lung zones, consistent with viral pneumonia, suggestive of COVID-19 infection.',
            'embedding': np.random.randn(64) * 0.3,
            'metadata': {
                'condition': 'covid',
                'severity': 'moderate',
                'findings': ['ground_glass_opacity', 'bilateral_involvement', 'peripheral_distribution'],
                'location': 'bilateral_peripheral'
            }
        },
        {
            'text': 'Extensive bilateral ground-glass and consolidative opacities with crazy-paving pattern, indicating severe COVID-19 pneumonia with ARDS features.',
            'embedding': np.random.randn(64) * 0.3,
            'metadata': {
                'condition': 'covid',
                'severity': 'severe',
                'findings': ['ground_glass_opacity', 'consolidation', 'crazy_paving', 'ARDS'],
                'location': 'bilateral_diffuse'
            }
        },
        {
            'text': 'Mild bilateral ground-glass opacities in lower lobes with subpleural sparing, early COVID-19 pneumonia presentation.',
            'embedding': np.random.randn(64) * 0.3,
            'metadata': {
                'condition': 'covid',
                'severity': 'mild',
                'findings': ['ground_glass_opacity', 'subpleural_sparing', 'lower_lobe'],
                'location': 'bilateral_lower'
            }
        },
        {
            'text': 'Consolidation with air bronchograms in the right lower lobe, suggestive of bacterial pneumonia rather than viral etiology.',
            'embedding': np.random.randn(64) * 0.3,
            'metadata': {
                'condition': 'pneumonia',
                'severity': 'moderate',
                'findings': ['consolidation', 'air_bronchograms'],
                'location': 'right_lower_lobe'
            }
        },
        {
            'text': 'Patchy bilateral infiltrates with reticular pattern and peripheral distribution, consistent with organizing pneumonia post-COVID-19.',
            'embedding': np.random.randn(64) * 0.3,
            'metadata': {
                'condition': 'covid',
                'severity': 'moderate',
                'findings': ['organizing_pneumonia', 'reticular_pattern', 'peripheral'],
                'location': 'bilateral'
            }
        },
        {
            'text': 'Pleural effusion bilaterally with basal atelectasis, may indicate cardiac or inflammatory etiology, follow-up recommended.',
            'embedding': np.random.randn(64) * 0.3,
            'metadata': {
                'condition': 'effusion',
                'severity': 'moderate',
                'findings': ['pleural_effusion', 'atelectasis'],
                'location': 'bilateral_basal'
            }
        },
        {
            'text': 'Cardiomegaly with vascular redistribution and Kerley B lines, suggesting pulmonary edema from cardiac origin.',
            'embedding': np.random.randn(64) * 0.3,
            'metadata': {
                'condition': 'cardiomegaly',
                'severity': 'moderate',
                'findings': ['enlarged_heart', 'pulmonary_edema', 'kerley_lines'],
                'location': 'cardiac'
            }
        },
        {
            'text': 'Resolution of previously noted ground-glass opacities, clear lung fields bilaterally indicating recovery from COVID-19 pneumonia.',
            'embedding': np.random.randn(64) * 0.3,
            'metadata': {
                'condition': 'normal',
                'severity': 'none',
                'findings': ['resolved', 'clear_lungs'],
                'location': 'bilateral'
            }
        },
        {
            'text': 'Multifocal ground-glass opacities with reverse halo sign, characteristic of organizing pneumonia, may be post-COVID sequelae.',
            'embedding': np.random.randn(64) * 0.3,
            'metadata': {
                'condition': 'covid',
                'severity': 'mild',
                'findings': ['ground_glass_opacity', 'reverse_halo_sign', 'organizing_pneumonia'],
                'location': 'multifocal'
            }
        }
    ]
    
    # Add entries to knowledge base
    texts = [entry['text'] for entry in sample_entries]
    embeddings = [entry['embedding'] for entry in sample_entries]
    metadatas = [entry['metadata'] for entry in sample_entries]
    
    kb.add_batch(texts=texts, embeddings=embeddings, metadatas=metadatas)
    
    print(f"✓ Populated knowledge base with {len(sample_entries)} sample medical cases")
    return len(sample_entries)


if __name__ == '__main__':
    print("LangChain RAG Module Test")
    print("=" * 60)
    
    # Test ChromaDB
    print("\n1. Testing ChromaDB Knowledge Base...")
    try:
        kb = ChromaDBMedicalKnowledgeBase()
        
        # Populate with sample data
        num_entries = populate_sample_knowledge_base(kb)
        
        # Test search
        query_emb = np.random.randn(64)
        results = kb.search(query_emb, top_k=3)
        print(f"  Search returned {len(results)} results")
        
        # Show stats
        stats = kb.get_stats()
        print(f"  Knowledge base stats: {stats}")
        
        print("  ✓ ChromaDB test passed")
    except Exception as e:
        print(f"  ✗ ChromaDB test failed: {e}")
    
    # Test Gemini (only if API key is set)
    print("\n2. Testing Gemini Explainer...")
    try:
        if RAGConfig.GEMINI_API_KEY and RAGConfig.GEMINI_API_KEY != 'your-gemini-api-key-here':
            explainer = GeminiMedicalExplainer()
            
            # Test explanation generation
            test_cases = [
                {
                    'text': 'Ground-glass opacities bilaterally',
                    'similarity': 0.95,
                    'metadata': {'condition': 'covid', 'severity': 'moderate'}
                }
            ]
            
            result = explainer.generate_explanation(
                prediction='COVID-19',
                confidence=0.89,
                retrieved_cases=test_cases
            )
            
            print(f"  Generated explanation: {result['explanation'][:100]}...")
            print("  ✓ Gemini test passed")
        else:
            print("  ⊗ Gemini API key not set, skipping test")
    except Exception as e:
        print(f"  ✗ Gemini test failed: {e}")
    
    # Test full pipeline
    print("\n3. Testing LangChain RAG Pipeline...")
    try:
        pipeline = LangChainRAGPipeline(knowledge_base=kb)
        
        # Test query
        query_emb = np.random.randn(64)
        result = pipeline.query(
            embedding=query_emb,
            prediction='COVID-19',
            confidence=0.87,
            top_k=3,
            generate_explanation=False  # Skip LLM for test
        )
        
        print(f"  Pipeline result keys: {list(result.keys())}")
        print(f"  Retrieved {len(result['retrieved_cases'])} cases")
        print(f"  Explanation hash: {result['explanation_hash'][:16]}...")
        print("  ✓ Pipeline test passed")
    except Exception as e:
        print(f"  ✗ Pipeline test failed: {e}")
    
    print("\n" + "=" * 60)
    print("LangChain RAG Module tests complete!")
