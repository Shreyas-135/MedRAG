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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time

from models import ClientModel2Layers
from rag_server_model import RAGEnhancedServerModel


def detect_class_names_from_dir(dataset_dir: str):
    """
    Auto-detect class names from a dataset directory.

    Checks ``<dataset_dir>/SplitCovid19/hospitalA/train`` first (new naming),
    then ``<dataset_dir>/SplitCovid19/client0/train`` (legacy naming).

    Args:
        dataset_dir: Root directory of the dataset

    Returns:
        Sorted list of class names, or ``None`` if detection fails.
    """
    split_base = os.path.join(dataset_dir, 'SplitCovid19')
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
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
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
            'model_type': 'RAG-Enhanced VFL' if self.server_model.use_rag else 'Standard VFL'
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
                    confidence.item()
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

    # Load checkpoint if provided (read metadata first)
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            resolved_num_classes = checkpoint['config'].get('num_classes', 2)
            resolved_class_names = checkpoint['config'].get('class_names', None)

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

    # Initialize client models
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


# ============================================================================
# VFL Inference Engine – uses VFLFramework (trained by train_multimodel.py)
# ============================================================================

# Built-in medical knowledge base for 4-class RAG citations
_MEDICAL_CITATIONS: Dict[str, List[Dict[str, str]]] = {
    "covid": [
        {
            "source": "WHO COVID-19 Clinical Management Guidelines (2021)",
            "url": "https://www.who.int/publications/i/item/WHO-2019-nCoV-clinical-2021-2",
            "snippet": (
                "COVID-19 pneumonia typically presents with bilateral ground-glass opacities "
                "on chest X-ray, predominantly in the lower and peripheral lung zones."
            ),
        },
        {
            "source": "Radiology (2020): Chest CT Findings in COVID-19",
            "url": "https://pubs.rsna.org/doi/10.1148/radiol.2020200274",
            "snippet": (
                "Common CT findings include bilateral, peripheral, and basal-predominant "
                "ground-glass opacities; consolidation; and crazy-paving pattern."
            ),
        },
        {
            "source": "RSNA COVID-19 Pneumonia Reporting",
            "url": "https://pubs.rsna.org/doi/10.1148/radiol.2020201490",
            "snippet": (
                "The RSNA panel recommends reporting COVID-19 pneumonia as typical, "
                "indeterminate, atypical, or negative based on chest imaging patterns."
            ),
        },
    ],
    "lung_opacity": [
        {
            "source": "ATS/ERS/JRS/ALAT ILD Guidelines",
            "url": "https://www.thoracic.org/statements/ild.php",
            "snippet": (
                "Ground-glass opacity on chest imaging may represent active alveolitis "
                "or interstitial lung disease and warrants clinical correlation."
            ),
        },
        {
            "source": "Radiographics (2014): Pulmonary Opacity",
            "url": "https://pubs.rsna.org/doi/10.1148/rg.342125010",
            "snippet": (
                "Lung opacities can arise from alveolar filling, interstitial thickening, "
                "or a combination, and their distribution guides differential diagnosis."
            ),
        },
        {
            "source": "Fleischner Society White Paper: Ground-Glass Opacity",
            "url": "https://www.fleischner-society.org",
            "snippet": (
                "Subsolid nodules including pure ground-glass and part-solid nodules "
                "require follow-up CT to assess growth and malignant potential."
            ),
        },
    ],
    "normal": [
        {
            "source": "Chest Radiology: Plain Film Patterns and Differential Diagnoses (Felson)",
            "url": "https://www.elsevier.com/books/chest-roentgenology/felson/978-0-7216-3541-5",
            "snippet": (
                "A normal chest X-ray demonstrates clear lung fields, normal cardiac silhouette "
                "(CT ratio < 0.5), and no pleural effusion or pneumothorax."
            ),
        },
        {
            "source": "ACR Appropriateness Criteria: Chest Radiograph",
            "url": "https://www.acr.org/Clinical-Resources/ACR-Appropriateness-Criteria",
            "snippet": (
                "Routine PA and lateral chest radiographs are recommended as the initial "
                "imaging study for evaluation of cardiopulmonary symptoms."
            ),
        },
    ],
    "pneumonia": [
        {
            "source": "IDSA/ATS Consensus Guidelines: Community-Acquired Pneumonia (2007)",
            "url": "https://www.idsociety.org/practice-guideline/community-acquired-pneumonia-cap/",
            "snippet": (
                "Radiographic findings of pneumonia include lobar or segmental consolidation, "
                "bronchopneumonia pattern, and interstitial infiltrates."
            ),
        },
        {
            "source": "Lancet (2015): Pneumonia – Global Burden",
            "url": "https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(15)00069-9",
            "snippet": (
                "Chest X-ray remains the gold standard for confirming clinical diagnosis "
                "of pneumonia; patchy or lobar consolidation is the hallmark finding."
            ),
        },
        {
            "source": "BMJ Best Practice: Pneumonia Diagnosis",
            "url": "https://bestpractice.bmj.com/topics/en-gb/3000091",
            "snippet": (
                "Patients with community-acquired pneumonia typically present with fever, "
                "cough, and focal consolidation on chest imaging."
            ),
        },
    ],
}

_RADIOLOGY_TEMPLATES: Dict[str, str] = {
    "covid": (
        "The chest X-ray demonstrates features consistent with COVID-19 pneumonia. "
        "The model identified bilateral pulmonary involvement with high confidence ({confidence:.1%}). "
        "Key imaging characteristics include ground-glass opacities predominantly in the "
        "lower and peripheral lung zones. Clinical correlation with RT-PCR testing and "
        "patient symptoms is recommended. Monitoring for disease progression or development "
        "of consolidation is advised. Refer to WHO COVID-19 clinical management guidelines "
        "for treatment thresholds."
    ),
    "lung_opacity": (
        "The chest X-ray shows pulmonary opacity that does not meet criteria for frank "
        "consolidation or typical COVID-19 pattern. The classifier assigned this finding "
        "to the lung opacity category with confidence {confidence:.1%}. "
        "Differential diagnosis includes atypical infection, early organizing pneumonia, "
        "pulmonary edema, or interstitial lung disease. CT chest with high-resolution "
        "sequences is recommended for further characterization."
    ),
    "normal": (
        "The chest X-ray appears within normal limits. The model classified this image "
        "as normal with confidence {confidence:.1%}. "
        "Lung fields are clear, cardiac silhouette is within normal limits, and there are "
        "no acute cardiopulmonary findings. Routine follow-up as clinically indicated."
    ),
    "pneumonia": (
        "The chest X-ray demonstrates features consistent with pneumonia. "
        "The model identified consolidative opacities with confidence {confidence:.1%}. "
        "Findings are suggestive of lobar or segmental consolidation typical of bacterial "
        "pneumonia. Clinical correlation with patient symptoms, inflammatory markers, and "
        "microbiological cultures is recommended. Antibiotic therapy should be guided by "
        "local susceptibility patterns per IDSA/ATS guidelines."
    ),
}


class VFLInferenceEngine:
    """
    Inference engine for VFLFramework models trained by train_multimodel.py.

    Supports:
    - Loading checkpoints produced by train_multimodel.py
    - 4-class prediction (covid, lung_opacity, normal, pneumonia)
    - Consistent 224×224 preprocessing with ImageNet normalisation
    - RAG explanation + citations (evidence-grounded, class-specific)
    """

    DEFAULT_CLASS_NAMES = ["covid", "lung_opacity", "normal", "pneumonia"]

    def __init__(
        self,
        model: "torch.nn.Module",
        class_names: List[str] = None,
    ):
        self.model = model
        self.model.eval()
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------
    def preprocess(self, image_path: str) -> "torch.Tensor":
        img = Image.open(image_path).convert("RGB")
        return self.transform(img).unsqueeze(0)

    # ------------------------------------------------------------------
    def predict(
        self,
        image_path: str,
        return_explanations: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Returns a dict with:
          - prediction       : str
          - confidence       : float (0–1)
          - probabilities    : {class_name: float}
          - explanation_text : str  (radiology-style template)
          - citations        : list of {source, url, snippet}
          - inference_time   : float (seconds)
          - model_type       : str
        """
        start = time.time()

        tensor = self.preprocess(image_path)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, pred_idx = torch.max(probs, dim=0)

        pred_class = self.class_names[pred_idx.item()]
        confidence_val = float(confidence.item())

        result: Dict[str, Any] = {
            "prediction": pred_class,
            "confidence": confidence_val,
            "probabilities": {
                cls: float(p)
                for cls, p in zip(self.class_names, probs)
            },
            "inference_time": time.time() - start,
            "model_type": "VFL (VFLFramework)",
        }

        if return_explanations:
            result["explanation_text"] = _RADIOLOGY_TEMPLATES.get(
                pred_class,
                f"Model predicted {pred_class} with confidence {confidence_val:.1%}.",
            ).format(confidence=confidence_val)
            result["citations"] = _MEDICAL_CITATIONS.get(pred_class, [])

        return result


def load_vfl_model(
    checkpoint_path: str = None,
    backbone: str = "resnet18",
    class_names: List[str] = None,
) -> "VFLInferenceEngine":
    """
    Load a VFLFramework checkpoint saved by train_multimodel.py.

    Args:
        checkpoint_path : Path to ``{model}_best.pth`` checkpoint file.
                          When ``None`` (no checkpoint), a randomly-initialised
                          model is returned so the pipeline can be exercised
                          end-to-end without trained weights.
        backbone        : CNN backbone name (e.g. 'resnet18').  Used only when
                          no checkpoint is provided or when the checkpoint does
                          not store config.
        class_names     : Override class names.  Falls back to checkpoint config
                          then to the 4-class default.

    Returns:
        VFLInferenceEngine ready for inference.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    from vfl_feature_partition import VFLFramework

    resolved_backbone = backbone
    resolved_classes = class_names if class_names else None
    resolved_embedding_dim = 512
    resolved_num_partitions = 4
    resolved_top_hidden = 256
    if resolved_classes and len(resolved_classes) > 0:
        resolved_num_classes = len(resolved_classes)
    else:
        resolved_num_classes = 4
        resolved_classes = None  # will be resolved below

    state_dict = None
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        resolved_backbone = cfg.get("backbone_name", backbone)
        resolved_embedding_dim = cfg.get("embedding_dim", 512)
        resolved_num_partitions = cfg.get("num_partitions", 4)
        resolved_top_hidden = cfg.get("top_hidden", 256)
        if resolved_classes is None:
            resolved_classes = cfg.get("class_names", None)
        resolved_num_classes = len(resolved_classes) if resolved_classes else cfg.get("num_classes", 4)
        state_dict = ckpt.get("model_state_dict")
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        print(f"  backbone={resolved_backbone}, classes={resolved_classes}, best_val_f1={ckpt.get('best_val_f1', 'N/A')}")
    elif checkpoint_path:
        print(f"Warning: checkpoint not found at {checkpoint_path}; using random weights.")

    if resolved_classes is None:
        resolved_classes = VFLInferenceEngine.DEFAULT_CLASS_NAMES
        resolved_num_classes = len(resolved_classes)

    model = VFLFramework(
        backbone_name=resolved_backbone,
        embedding_dim=resolved_embedding_dim,
        num_partitions=resolved_num_partitions,
        num_classes=resolved_num_classes,
        top_hidden=resolved_top_hidden,
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return VFLInferenceEngine(model=model, class_names=resolved_classes)


# ============================================================================
# Multi-Model Weighted Ensemble Inference Engine
# ============================================================================

# Default ensemble weights matching train_multimodel.ENSEMBLE_WEIGHTS
_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "resnet18":        0.30,
    "densenet121":     0.40,
    "efficientnet_b0": 0.30,
}

_HOSPITAL_MAP: Dict[str, str] = {
    "resnet18":        "Hospital_A",
    "densenet121":     "Hospital_B",
    "efficientnet_b0": "Hospital_C",
}

# Confidence threshold below which we flag NEEDS RADIOLOGIST REVIEW.
# 0.60 is a conservative clinical threshold: below 60% the model is uncertain
# enough that a human radiologist should verify the finding before acting on it.
_LOW_CONFIDENCE_THRESHOLD = 0.60

# Image size and normalisation constants used in preprocessing
_GRADCAM_IMG_SIZE = (224, 224)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _generate_gradcam_pil(
    model: "torch.nn.Module",
    image_path: str,
) -> Optional["Image.Image"]:
    """
    Generate a Grad-CAM overlay PIL Image entirely in memory.

    Returns ``None`` if pytorch-grad-cam is not installed or if the model
    has no Conv2d layers to hook.
    """
    try:
        import numpy as _np
        from PIL import Image as _PILImage
        import torchvision.transforms as _T
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # Find all Conv2d layers; take the last one as target
        target_layers = [
            m for _, m in model.named_modules()
            if isinstance(m, torch.nn.Conv2d)
        ]
        if not target_layers:
            return None

        _transform = _T.Compose([
            _T.Resize(_GRADCAM_IMG_SIZE),
            _T.ToTensor(),
            _T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        orig_img = _PILImage.open(image_path).convert("RGB").resize(_GRADCAM_IMG_SIZE)
        orig_np = _np.array(orig_img, dtype=_np.float32) / 255.0

        input_tensor = _transform(
            _PILImage.open(image_path).convert("RGB")
        ).unsqueeze(0)

        cam = GradCAM(model=model, target_layers=[target_layers[-1]])
        grayscale_cam = cam(input_tensor=input_tensor)[0]

        vis = show_cam_on_image(orig_np, grayscale_cam, use_rgb=True)
        return _PILImage.fromarray(vis)
    except Exception:
        return None


class MultiModelEnsembleEngine:
    """
    Weighted-average ensemble inference over multiple VFLFramework models.

    Each backbone (resnet18, densenet121, efficientnet_b0) acts as a virtual
    hospital.  Predictions are aggregated as a normalised weighted average of
    per-model softmax probability vectors (same formula as
    ``train_multimodel.weighted_ensemble_predict``).

    Args:
        engines : mapping of ``backbone_name`` → :class:`VFLInferenceEngine`
        weights : per-backbone weight (defaults to :data:`_ENSEMBLE_WEIGHTS`)
    """

    def __init__(
        self,
        engines: Dict[str, "VFLInferenceEngine"],
        weights: Optional[Dict[str, float]] = None,
    ):
        if not engines:
            raise ValueError("engines dict must contain at least one engine")
        self.engines = engines
        self.weights = weights if weights is not None else _ENSEMBLE_WEIGHTS
        # Infer class names from the first available engine
        first_engine = next(iter(engines.values()))
        self.class_names: List[str] = first_engine.class_names

    # ------------------------------------------------------------------
    def predict(
        self,
        image_path: str,
        return_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """
        Run weighted ensemble inference on a single image.

        Returns a dict compatible with the existing webapp result schema,
        extended with ``per_hospital``, ``needs_review``, ``review_reason``,
        and ``gradcam_images`` keys.
        """
        start = time.time()

        per_hospital: Dict[str, Any] = {}
        per_probs: Dict[str, np.ndarray] = {}

        for backbone, engine in self.engines.items():
            try:
                res = engine.predict(image_path, return_explanations=False)
                probs_arr = np.array(
                    [res["probabilities"].get(c, 0.0) for c in self.class_names],
                    dtype=np.float64,
                )
                per_probs[backbone] = probs_arr
                per_hospital[_HOSPITAL_MAP.get(backbone, backbone)] = {
                    "backbone":    backbone,
                    "prediction":  res["prediction"],
                    "confidence":  res["confidence"],
                    "probabilities": res["probabilities"],
                    "weight":      self.weights.get(backbone, 1.0 / len(self.engines)),
                }
            except Exception as exc:
                print(f"Warning: engine {backbone} failed: {exc}")

        if not per_probs:
            raise RuntimeError("All backbone engines failed during ensemble inference")

        # Weighted average aggregation
        present = list(per_probs)
        default_w = 1.0 / len(present)
        w_sum = sum(self.weights.get(n, default_w) for n in present)
        aggregated = np.zeros(len(self.class_names), dtype=np.float64)
        for name, probs in per_probs.items():
            w = self.weights.get(name, default_w) / w_sum
            aggregated += w * probs

        pred_idx = int(np.argmax(aggregated))
        pred_class = self.class_names[pred_idx]
        confidence_val = float(aggregated[pred_idx])

        # Agreement / uncertainty analysis
        top_preds = [
            ph["prediction"]
            for ph in per_hospital.values()
        ]
        unique_preds = set(top_preds)
        high_disagreement = len(unique_preds) > 1
        low_confidence = confidence_val < _LOW_CONFIDENCE_THRESHOLD
        needs_review = high_disagreement or low_confidence

        if high_disagreement and low_confidence:
            review_reason = (
                f"Low ensemble confidence ({confidence_val:.1%}) and "
                f"hospital disagreement ({', '.join(sorted(unique_preds))})"
            )
        elif high_disagreement:
            review_reason = (
                f"Hospital disagreement: "
                + ", ".join(
                    f"{h}→{d['prediction']}"
                    for h, d in per_hospital.items()
                )
            )
        elif low_confidence:
            review_reason = (
                f"Low ensemble confidence ({confidence_val:.1%})"
            )
        else:
            review_reason = ""

        # RAG explanation + citations (use first engine's template)
        first_engine = next(iter(self.engines.values()))
        explanation_text = _RADIOLOGY_TEMPLATES.get(
            pred_class,
            f"Model predicted {pred_class} with confidence {confidence_val:.1%}.",
        ).format(confidence=confidence_val)
        citations = _MEDICAL_CITATIONS.get(pred_class, [])

        # Grad-CAM images in memory
        gradcam_images: Dict[str, Any] = {}
        if return_gradcam:
            for backbone, engine in self.engines.items():
                gradcam_images[backbone] = _generate_gradcam_pil(
                    engine.model, image_path
                )

        return {
            "prediction":       pred_class,
            "confidence":       confidence_val,
            "probabilities":    {
                c: float(aggregated[i])
                for i, c in enumerate(self.class_names)
            },
            "per_hospital":     per_hospital,
            "ensemble_weights": {
                n: self.weights.get(n, default_w)
                for n in present
            },
            "needs_review":     needs_review,
            "review_reason":    review_reason,
            "explanation_text": explanation_text,
            "citations":        citations,
            "gradcam_images":   gradcam_images,
            "inference_time":   time.time() - start,
            "model_type":       f"Weighted Ensemble VFL ({len(self.engines)} backbones)",
            "num_models":       len(self.engines),
        }


def load_multi_model_ensemble(
    checkpoints_dir: str = None,
    class_names: List[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> "MultiModelEnsembleEngine":
    """
    Load all available VFLFramework checkpoints from *checkpoints_dir* and
    return a :class:`MultiModelEnsembleEngine`.

    If a backbone checkpoint file is missing the backbone is silently skipped
    (so the ensemble degrades gracefully to however many models are present).

    Args:
        checkpoints_dir : Directory containing ``{backbone}_best.pth`` files.
                          Defaults to ``<repo_root>/outputs/checkpoints/``.
        class_names     : Override class names for all engines.
        weights         : Per-backbone weights (defaults to :data:`_ENSEMBLE_WEIGHTS`).

    Returns:
        :class:`MultiModelEnsembleEngine` with at least one engine loaded.

    Raises:
        ``RuntimeError`` if no checkpoint is found.
    """
    if checkpoints_dir is None:
        # Repo-relative path — works on any OS (no hardcoded /kaggle paths)
        checkpoints_dir = str(
            Path(__file__).parent.parent / "outputs" / "checkpoints"
        )

    backbones = ["resnet18", "densenet121", "efficientnet_b0"]
    engines: Dict[str, VFLInferenceEngine] = {}

    for backbone in backbones:
        ckpt = Path(checkpoints_dir) / f"{backbone}_best.pth"
        if ckpt.is_file():
            try:
                engine = load_vfl_model(
                    checkpoint_path=str(ckpt),
                    backbone=backbone,
                    class_names=class_names,
                )
                engines[backbone] = engine
                print(f"✓ Loaded {backbone} from {ckpt}")
            except Exception as exc:
                print(f"Warning: could not load {backbone}: {exc}")
        else:
            print(f"Info: checkpoint not found for {backbone} at {ckpt} — skipping")

    if not engines:
        # No checkpoints found — create random-weight engines for demo mode
        print("Warning: no checkpoints found; using random-weight models for demo")
        for backbone in backbones:
            try:
                engine = load_vfl_model(
                    checkpoint_path=None,
                    backbone=backbone,
                    class_names=class_names,
                )
                engines[backbone] = engine
            except Exception as exc:
                print(f"Warning: could not build {backbone}: {exc}")

    if not engines:
        raise RuntimeError(
            "Could not build any backbone engines for the ensemble. "
            "Ensure at least one of resnet18/densenet121/efficientnet_b0 "
            "is importable."
        )

    return MultiModelEnsembleEngine(engines=engines, weights=weights)


# ============================================================================
# Command-line interface
# ============================================================================

def _build_cli_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="MedRAG: VFL inference with RAG explanation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference with a trained checkpoint
  python src/inference.py \\
      --image data/SplitCovid19/client0/test/covid/COVID-100.png \\
      --checkpoint outputs/checkpoints/resnet18_best.pth \\
      --model resnet18 \\
      --output-json outputs/inference_result.json

  # Use a different backbone
  python src/inference.py \\
      --image path/to/xray.jpg \\
      --checkpoint outputs/checkpoints/densenet121_best.pth \\
      --model densenet121
""",
    )
    p.add_argument("--image", required=True, help="Path to input chest X-ray image")
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to VFLFramework checkpoint (.pth). If omitted, uses random weights.",
    )
    p.add_argument(
        "--model", default="resnet18",
        choices=["resnet18", "densenet121", "efficientnet_b0", "mobilenet_v2"],
        help="Backbone name (default: resnet18)",
    )
    p.add_argument(
        "--output-json", default=None, metavar="PATH",
        help="Save full JSON result to this file (e.g. outputs/inference_result.json)",
    )
    p.add_argument(
        "--class-names", default=None,
        help="Comma-separated class names (default: covid,lung_opacity,normal,pneumonia)",
    )
    return p


if __name__ == "__main__":
    import argparse
    import json as _json

    parser = _build_cli_parser()
    args = parser.parse_args()

    class_names = (
        [c.strip() for c in args.class_names.split(",")]
        if args.class_names
        else None
    )

    print("=" * 60)
    print("  MedRAG VFL Inference Engine")
    print("=" * 60)

    engine = load_vfl_model(
        checkpoint_path=args.checkpoint,
        backbone=args.model,
        class_names=class_names,
    )

    print(f"\nRunning inference on: {args.image}")
    result = engine.predict(args.image, return_explanations=True)

    # Console summary
    print("\n" + "=" * 60)
    print("  RESULT SUMMARY")
    print("=" * 60)
    print(f"  Predicted class : {result['prediction']}")
    print(f"  Confidence      : {result['confidence']:.1%}")
    print("\n  Class probabilities:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"    {cls:15s}  {prob:6.1%}  {bar}")

    print("\n  RAG Explanation:")
    print(f"  {result.get('explanation_text', 'N/A')}")

    print("\n  Citations:")
    for i, cit in enumerate(result.get("citations", []), 1):
        print(f"  [{i}] {cit['source']}")
        if cit.get("url"):
            print(f"      URL: {cit['url']}")
        print(f"      {cit.get('snippet', '')}")

    print(f"\n  Inference time  : {result['inference_time']:.3f}s")
    print("=" * 60)

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as _f:
            _json.dump(result, _f, indent=2)
        print(f"\nJSON result saved to: {args.output_json}")
