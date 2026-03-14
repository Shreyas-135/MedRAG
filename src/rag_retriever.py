"""
RAG (Retrieval-Augmented Generation) Component for Medical Image Analysis
This module provides retrieval-augmented generation capabilities for X-ray image analysis
in the context of blockchain-enabled vertical federated learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import hashlib


class MedicalKnowledgeBase:
    """
    Medical knowledge base for storing and retrieving X-ray findings and diagnoses.
    Supports privacy-preserving retrieval for federated learning scenarios.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.knowledge_entries = []
        self.embeddings = []
        self.metadata = []
        
    def add_entry(self, text: str, embedding: np.ndarray, metadata: Dict):
        """
        Add a medical knowledge entry with its embedding and metadata.
        
        Args:
            text: Medical text description (e.g., diagnosis, findings)
            embedding: Feature embedding of the entry
            metadata: Additional metadata (condition, severity, etc.)
        """
        self.knowledge_entries.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve top-k most similar medical knowledge entries.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (text, similarity_score, metadata)
        """
        if len(self.embeddings) == 0:
            return []
            
        # Compute cosine similarity
        embeddings_array = np.array(self.embeddings)
        query_norm = np.linalg.norm(query_embedding)
        embeddings_norm = np.linalg.norm(embeddings_array, axis=1)
        
        # Avoid division by zero
        valid_indices = embeddings_norm > 0
        if query_norm == 0 or not np.any(valid_indices):
            return []
            
        similarities = np.zeros(len(self.embeddings))
        similarities[valid_indices] = np.dot(embeddings_array[valid_indices], query_embedding) / (
            embeddings_norm[valid_indices] * query_norm
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                self.knowledge_entries[idx],
                float(similarities[idx]),
                self.metadata[idx]
            ))
            
        return results
    
    def get_hash(self) -> str:
        """Generate a hash of the knowledge base for verification on blockchain."""
        content = json.dumps({
            'entries': self.knowledge_entries,
            'metadata': self.metadata
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class RAGModule(nn.Module):
    """
    Retrieval-Augmented Generation module for medical image analysis.
    Enhances VFL predictions with retrieved medical knowledge.
    """
    
    def __init__(self, embedding_dim: int = 64, num_classes: int = 2):
        super(RAGModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Knowledge base
        self.knowledge_base = MedicalKnowledgeBase(embedding_dim)
        
        # Attention mechanism for combining retrieved knowledge
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
        
        # Context integration layer
        self.context_integration = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def retrieve_context(self, embeddings: torch.Tensor, top_k: int = 3) -> torch.Tensor:
        """
        Retrieve relevant medical context for given embeddings.
        
        Args:
            embeddings: Batch of image embeddings [batch_size, embedding_dim]
            top_k: Number of knowledge entries to retrieve
            
        Returns:
            Retrieved context embeddings [batch_size, embedding_dim]
        """
        batch_size = embeddings.size(0)
        context_embeddings = []
        
        for i in range(batch_size):
            query_emb = embeddings[i].detach().cpu().numpy()
            retrieved = self.knowledge_base.retrieve(query_emb, top_k=top_k)
            
            if len(retrieved) > 0:
                # Average the retrieved embeddings
                retrieved_embs = [self.knowledge_base.embeddings[
                    self.knowledge_base.knowledge_entries.index(entry[0])
                ] for entry in retrieved]
                context_emb = np.mean(retrieved_embs, axis=0)
            else:
                # If no retrieval, use zero vector
                context_emb = np.zeros(self.embedding_dim)
                
            context_embeddings.append(context_emb)
            
        context_tensor = torch.tensor(
            np.array(context_embeddings), 
            dtype=torch.float32,
            device=embeddings.device
        )
        
        return context_tensor
        
    def forward(self, embeddings: torch.Tensor, use_retrieval: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with retrieval-augmented generation.
        
        Args:
            embeddings: Input embeddings from VFL clients [batch_size, embedding_dim]
            use_retrieval: Whether to use retrieval augmentation
            
        Returns:
            Tuple of (predictions, retrieved_context)
        """
        if use_retrieval and len(self.knowledge_base.knowledge_entries) > 0:
            # Retrieve relevant medical knowledge
            context = self.retrieve_context(embeddings)
            
            # Apply attention between embeddings and context
            embeddings_expanded = embeddings.unsqueeze(1)  # [batch, 1, embed_dim]
            context_expanded = context.unsqueeze(1)  # [batch, 1, embed_dim]
            
            attended, _ = self.attention(embeddings_expanded, context_expanded, context_expanded)
            attended = attended.squeeze(1)  # [batch, embed_dim]
            
            # Integrate context with original embeddings
            combined = torch.cat([embeddings, attended], dim=1)
            enhanced_embeddings = self.context_integration(combined)
        else:
            enhanced_embeddings = embeddings
            context = None
            
        # Classification
        predictions = self.classifier(enhanced_embeddings)
        
        return predictions, context
    
    def populate_knowledge_base(self, xray_dataset_info: List[Dict]):
        """
        Populate knowledge base with X-ray dataset information.
        
        Args:
            xray_dataset_info: List of dictionaries containing:
                - 'text': Medical description/diagnosis
                - 'embedding': Feature vector
                - 'condition': Medical condition (e.g., 'pneumonia', 'covid', 'normal')
                - 'severity': Severity level if applicable
                - 'findings': List of clinical findings
        """
        for entry in xray_dataset_info:
            self.knowledge_base.add_entry(
                text=entry['text'],
                embedding=entry['embedding'],
                metadata={
                    'condition': entry.get('condition', 'unknown'),
                    'severity': entry.get('severity', 'unknown'),
                    'findings': entry.get('findings', [])
                }
            )
    
    def verify_knowledge_base(self) -> str:
        """
        Generate verification hash for blockchain storage.
        
        Returns:
            SHA-256 hash of the knowledge base
        """
        return self.knowledge_base.get_hash()


class VerifiableRAGIntegrator:
    """
    Integrates RAG with blockchain for verifiable retrieval in federated learning.
    """
    
    def __init__(self, rag_module: RAGModule, blockchain_integrator=None):
        self.rag_module = rag_module
        self.blockchain_integrator = blockchain_integrator
        self.retrieval_logs = []
        
    def log_retrieval(self, query_hash: str, results_hash: str, timestamp: str):
        """
        Log retrieval operations for blockchain verification.
        
        Args:
            query_hash: Hash of the query embedding
            results_hash: Hash of retrieved results
            timestamp: Timestamp of retrieval
        """
        log_entry = {
            'query_hash': query_hash,
            'results_hash': results_hash,
            'timestamp': timestamp,
            'kb_hash': self.rag_module.verify_knowledge_base()
        }
        self.retrieval_logs.append(log_entry)
        
    def get_retrieval_proof(self) -> Dict:
        """
        Generate cryptographic proof of retrieval operations.
        
        Returns:
            Dictionary containing retrieval proof for blockchain verification
        """
        if len(self.retrieval_logs) == 0:
            return {'proof': None, 'logs': []}
            
        # Create Merkle-like proof structure
        logs_str = json.dumps(self.retrieval_logs, sort_keys=True)
        proof_hash = hashlib.sha256(logs_str.encode()).hexdigest()
        
        return {
            'proof': proof_hash,
            'logs': self.retrieval_logs,
            'kb_hash': self.rag_module.verify_knowledge_base()
        }
        
    def store_proof_on_blockchain(self, proof: Dict) -> bool:
        """
        Store retrieval proof on blockchain for verification.
        
        Args:
            proof: Retrieval proof dictionary
            
        Returns:
            Success status
        """
        if self.blockchain_integrator is None:
            print("Warning: No blockchain integrator available")
            return False
            
        # In practice, this would interact with a smart contract
        # to store the proof hash on blockchain
        print(f"Storing RAG proof on blockchain: {proof['proof']}")
        return True


def create_sample_xray_knowledge_base() -> List[Dict]:
    """
    Create a sample medical knowledge base for X-ray analysis.
    This would typically be populated from a curated medical database.
    
    Returns:
        List of knowledge base entries
    """
    sample_entries = [
        {
            'text': 'Normal chest X-ray showing clear lung fields with no signs of consolidation, effusion, or pneumothorax',
            'embedding': np.random.randn(64) * 0.1,  # Placeholder - would be actual medical image features
            'condition': 'normal',
            'severity': 'none',
            'findings': ['clear_lungs', 'normal_cardiac_silhouette']
        },
        {
            'text': 'Bilateral ground-glass opacities consistent with viral pneumonia, possibly COVID-19',
            'embedding': np.random.randn(64) * 0.1,
            'condition': 'covid',
            'severity': 'moderate',
            'findings': ['ground_glass_opacity', 'bilateral_involvement']
        },
        {
            'text': 'Consolidation in right lower lobe suggestive of bacterial pneumonia',
            'embedding': np.random.randn(64) * 0.1,
            'condition': 'pneumonia',
            'severity': 'moderate',
            'findings': ['consolidation', 'right_lower_lobe']
        },
        {
            'text': 'Pleural effusion in left hemithorax with blunting of costophrenic angle',
            'embedding': np.random.randn(64) * 0.1,
            'condition': 'effusion',
            'severity': 'mild',
            'findings': ['pleural_effusion', 'left_sided']
        },
        {
            'text': 'Cardiomegaly with increased cardiac-thoracic ratio suggesting heart failure',
            'embedding': np.random.randn(64) * 0.1,
            'condition': 'cardiomegaly',
            'severity': 'moderate',
            'findings': ['enlarged_heart', 'increased_ct_ratio']
        }
    ]
    return sample_entries
# ============================================================================
# Multi-Modal Knowledge Base (Added for 2025 Upgrade)
# ============================================================================

class MultiModalKnowledgeBase(MedicalKnowledgeBase):
    """Enhanced knowledge base with text and image embeddings"""
    
    def __init__(self, embedding_dim=64):
        super().__init__(embedding_dim)
        self.text_embeddings = []  # For clinical text descriptions
        self.image_embeddings = []  # For reference X-ray features
        
    def add_entry_multimodal(self, text, image_embedding, text_embedding, metadata):
        """
        Add entry with both image and text embeddings
        
        Args:
            text: Medical text description
            image_embedding:  Visual features from X-ray
            text_embedding: Text embedding from BioBERT/clinical model
            metadata: Additional info (condition, severity, etc.)
        """
        self.knowledge_entries.append(text)
        self.image_embeddings.append(image_embedding)
        self.text_embeddings.append(text_embedding)
        self.metadata.append(metadata)
    
    def retrieve_multimodal(self, query_image_emb, query_text_emb=None, top_k=5, alpha=0.6):
        """
        Multi-modal retrieval combining image and text similarity
        
        Args:
            query_image_emb: Image embedding
            query_text_emb:  Optional text embedding
            top_k:  Number of results
            alpha: Weight for image similarity (1-alpha for text)
        
        Returns:
            List of (text, similarity, metadata) tuples
        """
        if len(self.image_embeddings) == 0:
            return []
        
        # Image similarity
        image_embs = np.array(self.image_embeddings)
        image_scores = self._cosine_similarity(query_image_emb, image_embs)
        
        # Text similarity (if available)
        if query_text_emb is not None and len(self.text_embeddings) > 0:
            text_embs = np.array(self.text_embeddings)
            text_scores = self._cosine_similarity(query_text_emb, text_embs)
            
            # Combine scores
            combined_scores = alpha * image_scores + (1 - alpha) * text_scores
        else:
            combined_scores = image_scores
        
        # Get top-k
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                self.knowledge_entries[idx],
                float(combined_scores[idx]),
                self.metadata[idx]
            ))
        
        return results
    
    def _cosine_similarity(self, query, embeddings):
        """Compute cosine similarity"""
        query_norm = np.linalg.norm(query)
        emb_norms = np.linalg.norm(embeddings, axis=1)
        
        if query_norm == 0 or np.any(emb_norms == 0):
            return np.zeros(len(embeddings))
        
        similarities = np.dot(embeddings, query) / (emb_norms * query_norm)
        return similarities


def create_enhanced_knowledge_base():
    """Create knowledge base with multi-modal entries"""
    kb = MultiModalKnowledgeBase(embedding_dim=64)
    
    # Sample entries with both text and embeddings
    # In practice, these would come from a medical database
    entries = [
        {
            'text': 'Bilateral ground-glass opacities in peripheral and lower lung distribution, consistent with COVID-19 pneumonia',
            'image_emb': np.random.randn(64) * 0.5,  # Replace with actual image features
            'text_emb':  np.random.randn(64) * 0.5,   # Replace with BioBERT embeddings
            'metadata': {'condition': 'covid', 'severity': 'moderate', 'location': 'bilateral', 'pattern': 'ground-glass'}
        },
        {
            'text': 'Clear lung fields bilaterally with no focal consolidation, infiltrate, or pleural effusion',
            'image_emb': np.random.randn(64) * 0.5,
            'text_emb': np.random.randn(64) * 0.5,
            'metadata': {'condition':  'normal', 'severity': 'none', 'location': 'bilateral', 'pattern': 'clear'}
        },
        {
            'text': 'Patchy consolidation with air bronchograms in the right lower lobe suggesting bacterial pneumonia',
            'image_emb': np.random.randn(64) * 0.5,
            'text_emb': np.random.randn(64) * 0.5,
            'metadata': {'condition': 'pneumonia', 'severity': 'moderate', 'location': 'right-lower', 'pattern': 'consolidation'}
        },
        {
            'text':  'Reticular opacities with peripheral distribution and subpleural sparing, viral pneumonia pattern',
            'image_emb': np.random.randn(64) * 0.5,
            'text_emb':  np.random.randn(64) * 0.5,
            'metadata': {'condition': 'covid', 'severity': 'mild', 'location': 'peripheral', 'pattern': 'reticular'}
        },
        {
            'text':  'Diffuse bilateral infiltrates with crazy-paving pattern, severe COVID-19 presentation',
            'image_emb': np.random.randn(64) * 0.5,
            'text_emb':  np.random.randn(64) * 0.5,
            'metadata': {'condition': 'covid', 'severity': 'severe', 'location': 'bilateral', 'pattern': 'crazy-paving'}
        },
    ]
    
    for entry in entries:
        kb.add_entry_multimodal(
            entry['text'],
            entry['image_emb'],
            entry['text_emb'],
            entry['metadata']
        )
    
    print(f"✓ Multi-modal knowledge base created with {len(entries)} entries")
    return kb
