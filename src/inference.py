"""
Inference Module for MedRAG
Provides single image prediction with RAG explanations.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any
import time

from models import ClientModel2Layers, create_client_model
from rag_server_model import RAGEnhancedServerModel


def detect_class_names_from_dir(dataset_dir: str):
    """
    Auto-detect class names from a dataset directory.

    Checks the following locations in order (supports both the case where
    ``dataset_dir`` is the *parent* of ``SplitCovid19`` and the case where
    ``dataset_dir`` IS the ``SplitCovid19`` directory itself):

    1. ``<dataset_dir>/SplitCovid19/hospitalA/train``
    2. ``<dataset_dir>/SplitCovid19/client0/train``
    3. ``<dataset_dir>/hospitalA/train``  (dataset_dir == SplitCovid19)
    4. ``<dataset_dir>/client0/train``    (dataset_dir == SplitCovid19)

    Args:
        dataset_dir: Root directory of the dataset

    Returns:
        Sorted list of class names, or ``None`` if detection fails.
    """
    candidate_roots = [
        os.path.join(dataset_dir, 'SplitCovid19'),  # parent dir supplied
        dataset_dir,                                  # SplitCovid19 itself supplied
    ]
    for split_base in candidate_roots:
        for candidate in ['hospitalA', 'client0']:
            train_dir = os.path.join(split_base, candidate, 'train')
            if os.path.isdir(train_dir):
                classes = sorted([
                    d for d in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, d))
                ])
                if classes:
                    return classes
    return None


class MedRAGInference:
    """
    Inference engine for MedRAG federated learning system.
    
    Supports:
    - Single image prediction
    - RAG-based explanations
    - Confidence scoring
    - Medical guideline retrieval
    """
    
    def __init__(self, client_models: List[nn.Module], 
                 server_model: RAGEnhancedServerModel,
                 class_names: List[str] = None,
                 use_langchain: bool = False):
        """
        Initialize inference engine.
        
        Args:
            client_models: List of client models for feature extraction
            server_model: RAG-enhanced server model
            class_names: Names of output classes (default: ['Normal', 'COVID-19'])
            use_langchain: Whether to use LangChain for explanations
        """
        self.client_models = client_models
        self.server_model = server_model
        self.class_names = class_names or ['Normal', 'COVID-19']
        self.use_langchain = use_langchain
        
        # Set to evaluation mode
        for model in self.client_models:
            model.eval()
        self.server_model.eval()
        
        # Image preprocessing (ImageNet normalization matches pretrained backbones)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features using client models.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Aggregated embeddings from all clients
        """
        embeddings = []
        
        with torch.no_grad():
            for client_model in self.client_models:
                embedding = client_model(image_tensor)
                embeddings.append(embedding)
        
        # Sum embeddings (mimicking VFL aggregation)
        aggregated = torch.sum(torch.stack(embeddings), dim=0)
        return aggregated
    
    def get_rag_explanations(self, embeddings: torch.Tensor, 
                            top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get RAG-based explanations for the prediction.
        
        Args:
            embeddings: Feature embeddings
            top_k: Number of top explanations to retrieve
            
        Returns:
            List of explanation dictionaries
        """
        if not hasattr(self.server_model, 'use_rag') or not self.server_model.use_rag:
            return []
        
        rag_module = self.server_model.get_rag_module()
        if rag_module is None:
            return []
        
        # Get embedding as numpy for RAG retrieval
        embedding_np = embeddings[0].cpu().numpy()
        
        # Retrieve from knowledge base
        knowledge_base = rag_module.knowledge_base
        results = knowledge_base.retrieve(embedding_np, top_k=top_k)
        
        explanations = []
        for text, similarity, metadata in results:
            explanations.append({
                'text': text,
                'similarity': float(similarity),
                'condition': metadata.get('condition', 'Unknown'),
                'severity': metadata.get('severity', 'Unknown')
            })
        
        return explanations
    
    def predict(self, image_path: str, 
                return_explanations: bool = True,
                use_llm_explanation: bool = None) -> Dict[str, Any]:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to X-ray image
            return_explanations: Whether to include RAG explanations
            use_llm_explanation: Whether to use LLM for detailed explanations (uses self.use_langchain if None)
            
        Returns:
            Dictionary containing:
            - prediction: Predicted class name
            - confidence: Confidence score (0-1)
            - probabilities: Dict of class probabilities
            - inference_time: Time taken for inference (seconds)
            - explanations: RAG-based explanations (if enabled)
            - guidelines: Medical guidelines retrieved
            - rag_explanation: LLM-generated explanation (if LangChain enabled)
            - citations: Literature references (if LangChain enabled)
            - explanation_hash: Blockchain hash for audit trail (if LangChain enabled)
        """
        start_time = time.time()
        
        # Use class default if not specified
        if use_llm_explanation is None:
            use_llm_explanation = self.use_langchain
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Extract features
        embeddings = self.extract_features(image_tensor)
        
        # Get RAG explanations before prediction
        explanations = []
        guidelines = []
        if return_explanations:
            explanations = self.get_rag_explanations(embeddings, top_k=3)
            guidelines = [exp['text'] for exp in explanations]
        
        # Make prediction
        with torch.no_grad():
            outputs = self.server_model(embeddings)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)
        
        # Prepare base result
        result = {
            'prediction': self.class_names[predicted_idx.item()],
            'confidence': float(confidence.item()),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, probabilities)
            },
            'inference_time': time.time() - start_time,
            'model_type': 'RAG-Enhanced VFL' if self.server_model.use_rag else 'Standard VFL',
            'server_model_use_rag': bool(self.server_model.use_rag),
        }
        
        if return_explanations:
            result['explanations'] = explanations
            result['guidelines'] = guidelines
        
        # Add LLM explanation if requested
        if use_llm_explanation and hasattr(self.server_model, 'explain_prediction'):
            try:
                llm_result = self.server_model.explain_prediction(
                    embeddings, 
                    predicted_idx.item(), 
                    confidence.item(),
                    class_names=self.class_names
                )
                if llm_result:
                    result['rag_explanation'] = llm_result.get('rag_explanation', '')
                    result['citations'] = llm_result.get('citations', [])
                    result['explanation_hash'] = llm_result.get('explanation_hash', '')
                    result['retrieved_cases'] = llm_result.get('retrieved_cases', [])
            except Exception as e:
                print(f"Warning: Could not generate LLM explanation: {e}")
                result['rag_explanation'] = f"LLM explanation unavailable: {str(e)}"
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results
    
    def explain_features(self, image_path: str) -> Dict[str, Any]:
        """
        Extract and explain important features.
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary with feature analysis
        """
        image_tensor = self.preprocess_image(image_path)
        
        # Get embeddings from each client
        client_embeddings = []
        with torch.no_grad():
            for i, client_model in enumerate(self.client_models):
                embedding = client_model(image_tensor)
                client_embeddings.append({
                    f'client_{i}': {
                        'mean': float(embedding.mean()),
                        'std': float(embedding.std()),
                        'min': float(embedding.min()),
                        'max': float(embedding.max())
                    }
                })
        
        return {
            'client_features': client_embeddings,
            'feature_dimension': image_tensor.shape
        }


def load_inference_model(checkpoint_path: str = None,
                        use_rag: bool = True,
                        use_langchain: bool = False,
                        num_clients: int = 4,
                        class_names: List[str] = None,
                        dataset_dir: str = None) -> MedRAGInference:
    """
    Load models for inference.
    
    Args:
        checkpoint_path: Optional path to model checkpoint
        use_rag: Whether to use RAG enhancement
        use_langchain: Whether to use LangChain RAG pipeline
        num_clients: Number of client models
        class_names: Optional explicit list of class names. Takes precedence over
                     checkpoint metadata and ``dataset_dir`` detection.
        dataset_dir: Optional dataset root directory used to auto-detect class
                     names when ``class_names`` is not provided and the
                     checkpoint does not contain class metadata.
        
    Returns:
        Configured MedRAGInference instance
    """
    # Determine class names and num_classes.
    # Priority: explicit argument > checkpoint metadata > dataset_dir detection > default
    resolved_class_names = None
    resolved_num_classes = 2
    resolved_model_type = 'resnet_vgg'

    # Load checkpoint if provided (read metadata first)
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            resolved_num_classes = checkpoint['config'].get('num_classes', 2)
            resolved_class_names = checkpoint['config'].get('class_names', None)
            resolved_model_type = checkpoint['config'].get('model_type', 'resnet_vgg')

    # Explicit argument overrides checkpoint metadata
    if class_names is not None:
        resolved_class_names = class_names
        resolved_num_classes = len(class_names)
    elif resolved_class_names is not None:
        resolved_num_classes = len(resolved_class_names)
    elif dataset_dir is not None:
        detected = detect_class_names_from_dir(dataset_dir)
        if detected:
            resolved_class_names = detected
            resolved_num_classes = len(detected)

    # Fall back to binary default for backward compatibility
    if resolved_class_names is None:
        resolved_class_names = ['Normal', 'COVID-19']
        resolved_num_classes = 2

    # Defensive: ensure num_classes stays consistent with class_names list
    if len(resolved_class_names) != resolved_num_classes:
        resolved_num_classes = len(resolved_class_names)

    # Initialize client models using the same architecture as training
    client_models = []
    try:
        for _ in range(num_clients):
            model = create_client_model(model_type=resolved_model_type)
            model.eval()
            client_models.append(model)
    except Exception:
        # Fall back to default 2-layer model if factory fails
        client_models = []
        for _ in range(num_clients):
            model = ClientModel2Layers()
            model.eval()
            client_models.append(model)
    
    # Initialize server model
    server_model = RAGEnhancedServerModel(
        embedding_dim=64,
        num_classes=resolved_num_classes,
        use_rag=use_rag,
        use_langchain=use_langchain
    )
    server_model.eval()
    
    # Apply checkpoint weights if available
    if checkpoint is not None:
        if 'server_state_dict' in checkpoint:
            server_model.load_state_dict(checkpoint['server_state_dict'])
        elif 'model_state_dict' in checkpoint:
            server_model.load_state_dict(checkpoint['model_state_dict'])
        for i, client_model in enumerate(client_models):
            if f'client_{i}_state_dict' in checkpoint:
                client_model.load_state_dict(checkpoint[f'client_{i}_state_dict'])
    
    # Create inference engine
    inference = MedRAGInference(
        client_models=client_models,
        server_model=server_model,
        class_names=resolved_class_names,
        use_langchain=use_langchain
    )
    
    return inference


if __name__ == "__main__":
    # Example usage
    print("MedRAG Inference Engine Demo")
    print("=" * 50)
    
    # Load inference model
    print("Loading models...")
    inference = load_inference_model(use_rag=True, num_clients=4)
    print("✓ Models loaded successfully")
    
    print(f"\nInference Configuration:")
    print(f"  - Model type: {inference.server_model.__class__.__name__}")
    print(f"  - RAG enabled: {inference.server_model.use_rag}")
    print(f"  - Number of clients: {len(inference.client_models)}")
    print(f"  - Classes: {inference.class_names}")
    
    # Note: Actual prediction requires an image file
    print("\nReady for inference!")
    print("Use: inference.predict('path/to/xray.jpg')")
# ============================================================================
# Explainable Inference Engine (Added for 2025 Upgrade)
# ============================================================================

class ExplainableInferenceEngine(MedRAGInference):
    """Inference with explainability and uncertainty"""
    
    def __init__(self, client_models, server_model, model_types=None, 
                 enable_uncertainty=True, class_names=None):
        super().__init__(client_models, server_model, class_names)
        
        # Initialize explainability
        try:
            from explainability import ExplainabilityEngine
            self.explainers = []
            if model_types is None:
                model_types = ['cnn'] * len(self.client_models)
            
            for model, model_type in zip(self.client_models, model_types):
                explainer = ExplainabilityEngine(model, model_type=model_type)
                self.explainers.append(explainer)
            
            print(f"✓ Explainability enabled for {len(self.explainers)} models")
        except ImportError:
            print("Warning: explainability.py not found. XAI features disabled.")
            self.explainers = []
        
        # Enable uncertainty if requested
        if enable_uncertainty and hasattr(self.server_model, 'enable_uncertainty'):
            self.server_model.enable_uncertainty(num_samples=20)
    
    def predict_with_explanation(self, image_path, return_visualization=True):
        """
        Complete prediction with explanation and uncertainty
        
        Args:
            image_path: Path to X-ray image
            return_visualization: Whether to generate heatmap overlay
        
        Returns:
            Comprehensive result dictionary
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Extract features
        embeddings = self.extract_features(image_tensor)
        
        # Get prediction with uncertainty
        if hasattr(self.server_model, 'forward_with_uncertainty'):
            predictions, uncertainty = self.server_model.forward_with_uncertainty(embeddings)
        else:
            predictions = self.server_model(embeddings)
            uncertainty = None
        
        # Get predicted class and confidence
        probs = torch.softmax(predictions, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        
        # Get RAG explanations
        rag_explanations = self.get_rag_explanations(embeddings, top_k=3)
        
        # Generate visual explanations
        visual_explanations = []
        if self.explainers:
            for explainer in self.explainers:
                try:
                    explanation = explainer.explain(image_tensor, target_class=predicted_class.item())
                    visual_explanations.append(explanation)
                except Exception as e:
                    print(f"Warning: Could not generate explanation:  {e}")
                    visual_explanations.append({'error': str(e)})
        
        # Generate visualization if requested
        visualizations = []
        if return_visualization and self.explainers:
            for explainer, explanation in zip(self.explainers, visual_explanations):
                try:
                    if 'error' not in explanation:
                        viz = explainer.visualize(image_path, explanation)
                        visualizations.append(viz)
                except Exception as e:
                    print(f"Warning: Could not create visualization: {e}")
        
        # Check if should defer to human
        defer_decision = False
        defer_reason = "N/A"
        if uncertainty is not None and hasattr(self.server_model, 'should_defer'):
            defer_decision, defer_reason = self.server_model.should_defer(uncertainty, threshold=0.7)
        
        # Generate textual explanation
        from explainability import generate_textual_explanation
        text_explanation = generate_textual_explanation(
            predicted_class.item(),
            rag_explanations,
            confidence.item() * 100,
            explanation_type=visual_explanations[0].get('type', 'unknown') if visual_explanations else 'none'
        )
        
        # Compile results
        result = {
            'prediction': self.class_names[predicted_class.item()],
            'predicted_class': predicted_class.item(),
            'confidence': confidence.item() * 100,
            'probabilities': probs[0].cpu().numpy().tolist(),
            'rag_explanations': rag_explanations,
            'visual_explanations': visual_explanations,
            'visualizations': visualizations,
            'uncertainty': uncertainty,
            'defer_to_human': defer_decision,
            'defer_reason': defer_reason,
            'text_explanation': text_explanation
        }
        
        return result
    
    def batch_predict_with_explanation(self, image_paths, save_dir=None):
        """
        Batch prediction with explanations
        
        Args: 
            image_paths: List of image paths
            save_dir: Directory to save visualizations (optional)
        
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.predict_with_explanation(image_path, return_visualization=(save_dir is not None))
            
            # Save visualizations if requested
            if save_dir and result['visualizations']:
                import os
                os.makedirs(save_dir, exist_ok=True)
                for j, viz in enumerate(result['visualizations']):
                    output_path = os.path.join(save_dir, f"explanation_{i}_model_{j}.png")
                    viz.save(output_path)
            
            results.append(result)
        
        return results
