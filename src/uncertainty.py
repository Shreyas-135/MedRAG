"""
Uncertainty Quantification for Federated Medical Imaging
Implements Monte Carlo Dropout and ensemble methods
"""

import torch
import torch.nn as nn
import numpy as np


class UncertaintyQuantifier:
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, model, num_samples=20):
        self.model = model
        self.num_samples = num_samples
    
    def predict_with_uncertainty(self, x):
        """
        Generate predictions with uncertainty estimates
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary with mean prediction, std, and confidence
        """
        # Enable dropout at inference
        self.model.train()
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(torch.softmax(pred, dim=1))
        
        # Stack predictions
        predictions = torch.stack(predictions)  # [num_samples, batch_size, num_classes]
        
        # Calculate statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Uncertainty metrics
        # 1. Predictive entropy
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1)
        
        # 2. Mutual information (epistemic uncertainty)
        expected_entropy = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=2).mean(dim=0)
        mutual_info = entropy - expected_entropy
        
        # 3. Confidence (1 - normalized entropy)
        max_entropy = np.log(mean_pred.shape[1])
        confidence = 1 - (entropy / max_entropy)
        
        # Set back to eval mode
        self.model.eval()
        
        return {
            'prediction': mean_pred,
            'std': std_pred,
            'entropy': entropy,
            'mutual_information': mutual_info,
            'confidence': confidence,
            'all_predictions': predictions
        }
    
    def should_defer_to_human(self, uncertainty_result, threshold=0.7):
        """
        Determine if prediction should be deferred to human expert
        
        Args:
            uncertainty_result: Output from predict_with_uncertainty
            threshold:  Confidence threshold for deferral
        
        Returns:
            Boolean and reason string
        """
        confidence = uncertainty_result['confidence'].item()
        entropy = uncertainty_result['entropy'].item()
        mutual_info = uncertainty_result['mutual_information'].item()
        
        if confidence < threshold:
            return True, f"Low confidence ({confidence:.2f} < {threshold})"
        
        if entropy > 0.5:
            return True, f"High predictive uncertainty (entropy: {entropy:.2f})"
        
        if mutual_info > 0.3:
            return True, f"High model uncertainty (mutual info: {mutual_info:.2f})"
        
        return False, "Prediction is reliable"
