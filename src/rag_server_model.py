"""
RAG-Enhanced Server Model for Vertical Federated Learning
Integrates Retrieval-Augmented Generation with the existing VFL framework
"""

import torch
import torch.nn as nn
from rag_retriever import RAGModule, VerifiableRAGIntegrator, create_sample_xray_knowledge_base

# Import LangChain RAG components (optional)
try:
    from langchain_rag import LangChainRAGPipeline, ChromaDBMedicalKnowledgeBase, populate_sample_knowledge_base
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class RAGEnhancedServerModel(nn.Module):
    """
    Server model enhanced with RAG capabilities for medical image analysis.
    This model receives aggregated embeddings from VFL clients and uses
    RAG to retrieve relevant medical knowledge for improved predictions.
    """
    
    def __init__(self, embedding_dim: int = 64, num_classes: int = 2, use_rag: bool = True, use_langchain: bool = False):
        super(RAGEnhancedServerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.use_rag = use_rag
        self.use_langchain = use_langchain
        self.langchain_pipeline = None
        
        if self.use_rag:
            if self.use_langchain and LANGCHAIN_AVAILABLE:
                # Initialize LangChain RAG pipeline
                try:
                    kb = ChromaDBMedicalKnowledgeBase()
                    # Populate if empty
                    if kb.get_stats()['total_entries'] == 0:
                        populate_sample_knowledge_base(kb)
                    self.langchain_pipeline = LangChainRAGPipeline(knowledge_base=kb)
                    print("✓ Using LangChain RAG pipeline")
                except Exception as e:
                    print(f"Warning: Could not initialize LangChain RAG: {e}")
                    print("Falling back to simple RAG")
                    self.use_langchain = False
            
            if not self.use_langchain:
                # Initialize simple RAG module
                self.rag_module = RAGModule(embedding_dim=embedding_dim, num_classes=num_classes)
                
                # Initialize with sample medical knowledge
                # In practice, this would be populated with actual medical data
                sample_kb = create_sample_xray_knowledge_base()
                self.rag_module.populate_knowledge_base(sample_kb)
        else:
            # Fallback to standard classification
            self.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.BatchNorm1d(embedding_dim),
                nn.Linear(embedding_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )
    
    def forward(self, x):
        """
        Forward pass with optional RAG enhancement.
        
        Args:
            x: Aggregated embeddings from VFL clients [batch_size, embedding_dim]
            
        Returns:
            Predictions [batch_size, num_classes]
        """
        if self.use_rag:
            if self.use_langchain:
                # For LangChain, we still use the simple RAG module for forward pass
                # LangChain pipeline is used for explanation generation, not prediction
                # Fall back to simple classification for now
                # TODO: Integrate LangChain in forward pass if needed
                if hasattr(self, 'rag_module'):
                    predictions, context = self.rag_module(x, use_retrieval=True)
                    return predictions
                else:
                    # Create a simple classifier on the fly
                    if not hasattr(self, 'fc'):
                        self.fc = nn.Sequential(
                            nn.Dropout(0.4),
                            nn.BatchNorm1d(self.embedding_dim),
                            nn.Linear(self.embedding_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(128, self.num_classes),
                        ).to(x.device)
                    return self.fc(x)
            else:
                predictions, context = self.rag_module(x, use_retrieval=True)
                return predictions
        else:
            return self.fc(x)
    
    def get_rag_module(self):
        """Get the RAG module for external access."""
        if self.use_rag and not self.use_langchain:
            return self.rag_module
        return None
    
    def get_langchain_pipeline(self):
        """Get the LangChain RAG pipeline."""
        if self.use_langchain:
            return self.langchain_pipeline
        return None
    
    def explain_prediction(self, embedding, prediction, confidence):
        """
        Generate LLM explanation for a prediction.
        
        Args:
            embedding: Input embedding tensor
            prediction: Predicted class
            confidence: Prediction confidence
            
        Returns:
            Explanation dictionary or None
        """
        if self.use_langchain and self.langchain_pipeline:
            # Convert to numpy
            import numpy as np
            emb_np = embedding.detach().cpu().numpy()
            if emb_np.ndim > 1:
                emb_np = emb_np[0]  # Get first sample
            
            # Query pipeline
            class_names = ['Normal', 'COVID-19']
            pred_name = class_names[prediction] if prediction < len(class_names) else str(prediction)
            
            result = self.langchain_pipeline.query(
                embedding=emb_np,
                prediction=pred_name,
                confidence=confidence,
                generate_explanation=True
            )
            return result
        return None
    
    def retrieve_with_citations(self, embedding, top_k=5):
        """
        Retrieve medical literature references with citations.
        
        Args:
            embedding: Input embedding tensor
            top_k: Number of references to retrieve
            
        Returns:
            List of retrieved cases with citations
        """
        if self.use_langchain and self.langchain_pipeline:
            import numpy as np
            emb_np = embedding.detach().cpu().numpy()
            if emb_np.ndim > 1:
                emb_np = emb_np[0]
            
            kb = self.langchain_pipeline.knowledge_base
            results = kb.search(emb_np, top_k=top_k)
            return results
        elif self.use_rag and hasattr(self, 'rag_module'):
            # Use simple RAG retrieval
            import numpy as np
            emb_np = embedding.detach().cpu().numpy()
            if emb_np.ndim > 1:
                emb_np = emb_np[0]
            
            kb = self.rag_module.knowledge_base
            results = kb.retrieve(emb_np, top_k=top_k)
            # Format to match LangChain output
            formatted = []
            for text, similarity, metadata in results:
                formatted.append({
                    'text': text,
                    'similarity': similarity,
                    'metadata': metadata
                })
            return formatted
        return []
    
    def get_chromadb_stats(self):
        """
        Get ChromaDB statistics.
        
        Returns:
            Statistics dictionary
        """
        if self.use_langchain and self.langchain_pipeline:
            return self.langchain_pipeline.knowledge_base.get_stats()
        return None
    
    def update_knowledge_base(self, new_entries):
        """
        Update the medical knowledge base with new entries.
        
        Args:
            new_entries: List of new knowledge base entries
        """
        if self.use_rag:
            self.rag_module.populate_knowledge_base(new_entries)
    
    def verify_rag_integrity(self):
        """
        Verify the integrity of the RAG knowledge base.
        
        Returns:
            Hash of the knowledge base for blockchain verification
        """
        if self.use_rag:
            return self.rag_module.verify_knowledge_base()
        return None


class BlockchainRAGVFLSystem:
    """
    Complete system integrating Blockchain, VFL, and RAG for
    privacy-preserving cross-hospital medical imaging.
    """
    
    def __init__(self, num_clients: int, blockchain_integrator=None, use_rag: bool = True):
        self.num_clients = num_clients
        self.blockchain_integrator = blockchain_integrator
        self.use_rag = use_rag
        
        # Initialize client models (feature extractors)
        self.client_models = []
        
        # Initialize RAG-enhanced server model
        self.server_model = RAGEnhancedServerModel(
            embedding_dim=64,
            num_classes=2,
            use_rag=use_rag
        )
        
        # Initialize RAG integrator for blockchain verification
        if use_rag:
            rag_module = self.server_model.get_rag_module()
            self.rag_integrator = VerifiableRAGIntegrator(
                rag_module=rag_module,
                blockchain_integrator=blockchain_integrator
            )
        else:
            self.rag_integrator = None
    
    def train_step(self, client_embeddings, labels, with_blockchain=False):
        """
        Perform one training step with optional blockchain verification.
        
        Args:
            client_embeddings: List of embeddings from each client
            labels: Ground truth labels
            with_blockchain: Whether to use blockchain for aggregation
            
        Returns:
            Loss value
        """
        # Aggregate embeddings (potentially through blockchain)
        if with_blockchain and self.blockchain_integrator:
            # Use blockchain for aggregation
            aggregated_embeddings = self._blockchain_aggregate(client_embeddings)
        else:
            # Direct aggregation
            aggregated_embeddings = torch.sum(torch.stack(client_embeddings), dim=0)
        
        # Forward pass through RAG-enhanced server
        predictions = self.server_model(aggregated_embeddings)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)
        
        return loss, predictions
    
    def _blockchain_aggregate(self, embeddings):
        """Aggregate embeddings through blockchain smart contract."""
        # This would interact with the blockchain integrator
        # For now, we do simple aggregation
        return torch.sum(torch.stack(embeddings), dim=0)
    
    def log_rag_retrieval(self, query_hash, results_hash, timestamp):
        """
        Log RAG retrieval operations for blockchain verification.
        
        Args:
            query_hash: Hash of query
            results_hash: Hash of results
            timestamp: Timestamp
        """
        if self.rag_integrator:
            self.rag_integrator.log_retrieval(query_hash, results_hash, timestamp)
    
    def get_rag_proof(self):
        """
        Get cryptographic proof of RAG operations.
        
        Returns:
            Proof dictionary
        """
        if self.rag_integrator:
            return self.rag_integrator.get_retrieval_proof()
        return None
    
    def store_rag_proof_on_blockchain(self):
        """
        Store RAG proof on blockchain for verification.
        
        Returns:
            Success status
        """
        if self.rag_integrator:
            proof = self.get_rag_proof()
            return self.rag_integrator.store_proof_on_blockchain(proof)
        return False
    
    def verify_system_integrity(self):
        """
        Verify the integrity of the entire RAG-VFL system.
        
        Returns:
            Dictionary with verification results
        """
        verification = {
            'server_model_hash': None,
            'rag_kb_hash': None,
            'rag_proof_hash': None,
            'verified': False
        }
        
        # Get RAG knowledge base hash
        if self.use_rag:
            verification['rag_kb_hash'] = self.server_model.verify_rag_integrity()
            
            # Get RAG proof hash
            proof = self.get_rag_proof()
            if proof:
                verification['rag_proof_hash'] = proof.get('proof')
        
        # System is verified if all hashes are present
        verification['verified'] = all([
            verification['rag_kb_hash'] is not None if self.use_rag else True,
            True  # Add more verification checks as needed
        ])
        
        return verification

# ============================================================================
# Uncertainty-Aware RAG Server (Added for 2025 Upgrade)
# ============================================================================

class UncertaintyAwareRAGServer(RAGEnhancedServerModel):
    """RAG-enhanced server with uncertainty quantification"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_quantifier = None
    
    def enable_uncertainty(self, num_samples=20):
        """Enable Monte Carlo Dropout uncertainty"""
        try:
            from uncertainty import UncertaintyQuantifier
            self.uncertainty_quantifier = UncertaintyQuantifier(self, num_samples=num_samples)
            print(f"✓ Uncertainty quantification enabled (MC samples: {num_samples})")
        except ImportError:
            print("Warning: uncertainty.py not found. Uncertainty quantification disabled.")
            self.uncertainty_quantifier = None
    
    def forward_with_uncertainty(self, x, use_retrieval=True):
        """
        Forward pass with uncertainty estimation
        
        Returns:
            Tuple of (predictions, uncertainty_metrics)
        """
        if self.uncertainty_quantifier is None:
            # Standard forward
            predictions = self.forward(x)
            return predictions, None
        
        # Get predictions with uncertainty
        uncertainty_result = self.uncertainty_quantifier.predict_with_uncertainty(x)
        
        return uncertainty_result['prediction'], uncertainty_result
    
    def should_defer(self, uncertainty_result, threshold=0.7):
        """Check if prediction should be deferred to human expert"""
        if self.uncertainty_quantifier is None: 
            return False, "Uncertainty not enabled"
        
        return self.uncertainty_quantifier.should_defer_to_human(uncertainty_result, threshold)
