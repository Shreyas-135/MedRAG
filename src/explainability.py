"""
Explainability Module for Medical Image Classification
Provides GradCAM for CNNs and Attention Visualization for ViTs
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not installed. Install with: pip install pytorch-grad-cam")


class ExplainabilityEngine: 
    """Unified explainability interface for CNNs and ViTs"""
    
    def __init__(self, model, model_type='cnn'):
        self.model = model
        self.model_type = model_type
        
        if model_type == 'cnn' and GRADCAM_AVAILABLE:
            self._init_gradcam()
    
    def _init_gradcam(self):
        """Initialize GradCAM for CNN models"""
        # Find the last convolutional layer
        target_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layers.append(module)
        
        if target_layers:
            self.grad_cam = GradCAM(model=self.model, target_layers=[target_layers[-1]])
        else:
            self.grad_cam = None
    
    def explain(self, image_tensor, target_class=None):
        """
        Generate explanation for prediction
        
        Args:
            image_tensor: Input image tensor [1, 3, H, W]
            target_class: Target class for explanation (None = predicted class)
        
        Returns:
            Dictionary with explanation data
        """
        if self.model_type == 'cnn':
            return self._explain_cnn(image_tensor, target_class)
        elif self.model_type == 'vit':
            return self._explain_vit(image_tensor, target_class)
        else:
            return {'error': 'Unknown model type'}
    
    def _explain_cnn(self, image_tensor, target_class=None):
        """Generate GradCAM explanation for CNN"""
        if not GRADCAM_AVAILABLE or not hasattr(self, 'grad_cam') or self.grad_cam is None:
            return {'error': 'GradCAM not available for this model'}
        
        # Generate CAM
        grayscale_cam = self.grad_cam(input_tensor=image_tensor, targets=target_class)
        grayscale_cam = grayscale_cam[0, :]  # First image in batch
        
        return {
            'type': 'gradcam',
            'heatmap': grayscale_cam,
            'method': 'GradCAM'
        }
    
    def _explain_vit(self, image_tensor, target_class=None):
        """Generate attention map explanation for ViT"""
        # Simplified attention extraction
        # In practice, you'd extract actual attention weights from ViT layers
        # This is a placeholder that generates a mock attention map
        
        # For now, return a simple gradient-based map
        self.model.eval()
        image_tensor.requires_grad = True
        
        output = self.model(image_tensor)
        if target_class is not None:
            score = output[0, target_class]
        else:
            score = output.max()
        
        score.backward()
        
        # Use gradients as proxy for attention
        gradients = image_tensor.grad[0].abs().mean(dim=0)
        attention_map = gradients.cpu().numpy()
        
        return {
            'type': 'attention',
            'heatmap': attention_map,
            'method': 'Gradient-based Attention'
        }
    
    def visualize(self, image_path, explanation, output_path=None):
        """
        Create visualization overlay
        
        Args:
            image_path: Path to original image
            explanation:  Explanation dictionary from explain()
            output_path: Path to save visualization (optional)
        
        Returns:
            PIL Image with overlay
        """
        # Load original image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image_np = np.array(image) / 255.0
        
        # Get heatmap
        heatmap = explanation.get('heatmap')
        if heatmap is None:
            return image
        
        # Resize heatmap to match image
        try:
            from scipy.ndimage import zoom
            if heatmap.shape != (224, 224):
                zoom_factors = (224 / heatmap.shape[0], 224 / heatmap.shape[1])
                heatmap = zoom(heatmap, zoom_factors, order=1)
        except ImportError:
            # Fallback to simple resize
            from PIL import Image as PILImage
            heatmap = PILImage.fromarray((heatmap * 255).astype(np.uint8))
            heatmap = np.array(heatmap.resize((224, 224))) / 255.0
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Create overlay
        if GRADCAM_AVAILABLE: 
            cam_image = show_cam_on_image(image_np, heatmap, use_rgb=True)
            cam_pil = Image.fromarray(cam_image)
        else:
            # Fallback:  simple overlay
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            
            heatmap_colored = cm.jet(heatmap)[:, :, :3]
            cam_image = (0.6 * image_np + 0.4 * heatmap_colored)
            cam_image = (cam_image * 255).astype(np.uint8)
            cam_pil = Image.fromarray(cam_image)
        
        if output_path:
            cam_pil.save(output_path)
        
        return cam_pil


def generate_textual_explanation(prediction, rag_results, confidence, explanation_type='gradcam'):
    """
    Generate human-readable explanation
    
    Args:
        prediction: Predicted class
        rag_results: RAG retrieval results
        confidence: Prediction confidence
        explanation_type: Type of visual explanation used
    
    Returns:
        Textual explanation string
    """
    class_names = ['Normal', 'COVID-19']
    predicted_class = class_names[prediction]
    
    explanation = f"**Prediction**:  {predicted_class} ({confidence:.1f}% confidence)\n\n"
    
    if explanation_type == 'gradcam':
        explanation += "**Visual Explanation**: The highlighted regions show areas that most influenced the AI's decision.  "
        explanation += "Brighter areas had stronger impact on the classification.\n\n"
    elif explanation_type == 'attention': 
        explanation += "**Visual Explanation**: The attention map shows which image regions the AI focused on during analysis. "
        explanation += "The model uses a transformer architecture that learns to attend to relevant anatomical features.\n\n"
    
    if rag_results:
        explanation += "**Supporting Medical Knowledge**:\n"
        for i, result in enumerate(rag_results[: 3], 1):
            finding = result.get('text', 'N/A')
            similarity = result.get('similarity', 0)
            explanation += f"{i}. {finding} (relevance: {similarity:.2f})\n"
    
    return explanation
