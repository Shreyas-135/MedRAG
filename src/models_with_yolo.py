"""
YOLO-Enhanced Models for MedRAG Vertical Federated Learning

This module extends the existing models.py with YOLO (You Only Look Once) 
architecture support for object detection-based X-ray feature extraction.

Features:
- YOLOv5 and YOLOv8 feature extraction
- Hybrid ResNet + YOLO architectures
- Compatible with VFL framework (64-dim embeddings)
- Maintains backward compatibility with existing models

Usage:
    from models_with_yolo import create_client_model
    
    # Create YOLO model
    model = create_client_model('yolo5', embedding_dim=64)
    
    # Create hybrid model
    model = create_client_model('resnet_yolo', embedding_dim=64)
"""

import torch
import torch.nn as nn
import torchvision

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")


class ClientModelYOLO(nn.Module):
    """YOLO-based client model for X-ray feature extraction.
    
    Uses YOLO architecture for feature extraction before the detection head,
    projecting to a fixed embedding dimension for VFL compatibility.
    """
    
    def __init__(self, embedding_dim=64, yolo_version='yolov5'):
        """
        Initialize YOLO-based client model.
        
        Args:
            embedding_dim: Output embedding dimension (default: 64)
            yolo_version: YOLO version to use ('yolov5' or 'yolov8')
        """
        super(ClientModelYOLO, self).__init__()
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package not available. Install with: pip install ultralytics")
        
        self.yolo_version = yolo_version
        self.embedding_dim = embedding_dim
        
        # Load YOLO model (use smallest variant for efficiency)
        if yolo_version == 'yolov5':
            # YOLOv5n (nano) - lightweight version
            try:
                self.yolo = YOLO('yolov5n.pt')
            except:
                # Fallback to yolov8n if yolov5 not available
                print("Warning: YOLOv5 not available, using YOLOv8 instead")
                self.yolo = YOLO('yolov8n.pt')
                self.yolo_version = 'yolov8'
        elif yolo_version == 'yolov8':
            # YOLOv8n (nano) - lightweight version
            self.yolo = YOLO('yolov8n.pt')
        else:
            raise ValueError(f"Unsupported YOLO version: {yolo_version}")
        
        # Extract the backbone (feature extractor) from YOLO
        # YOLO model structure: backbone -> neck -> head
        # Using first 10 layers as feature extractor (typically up to C3 module)
        # This can be adjusted based on YOLO version and desired feature level
        self.backbone_layers = 10  # Configurable for different YOLO versions
        self.backbone = self.yolo.model.model[:self.backbone_layers]
        
        # Freeze YOLO backbone (optional - can be unfrozen for fine-tuning)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Adaptive pooling to get fixed-size features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension after YOLO backbone (typically 256 for nano models)
        # This will be automatically detected
        self.feature_dim = None
        self._detect_feature_dim()
        
        # Projection head to match embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
            nn.Tanh(),  # Output in range [-1, 1] like original models
        )
    
    def _detect_feature_dim(self):
        """Detect feature dimension from YOLO backbone."""
        # Create dummy input to detect output dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            if isinstance(features, (list, tuple)):
                features = features[-1]  # Take last feature map
            pooled = self.adaptive_pool(features)
            self.feature_dim = pooled.view(pooled.size(0), -1).shape[1]
    
    def forward(self, x):
        """
        Forward pass through YOLO feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        # Extract features using YOLO backbone
        features = self.backbone(x)
        
        # Handle multiple feature maps (take the last one)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # Global average pooling
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        
        # Project to embedding dimension
        embedding = self.projection(features)
        
        return embedding


class ClientModelResNetYOLO(nn.Module):
    """Hybrid ResNet + YOLO client model for enhanced feature extraction.
    
    Combines traditional CNN features (ResNet) with object detection features (YOLO)
    for comprehensive X-ray analysis.
    """
    
    def __init__(self, embedding_dim=64, yolo_version='yolov5'):
        """
        Initialize hybrid ResNet + YOLO model.
        
        Args:
            embedding_dim: Output embedding dimension (default: 64)
            yolo_version: YOLO version to use ('yolov5' or 'yolov8')
        """
        super(ClientModelResNetYOLO, self).__init__()
        
        # ResNet50 feature extractor
        # Note: Using 'pretrained=True' for backward compatibility
        # For newer versions, use: torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # YOLO feature extractor
        self.yolo_extractor = ClientModelYOLO(
            embedding_dim=embedding_dim // 2,  # Half the embedding for each branch
            yolo_version=yolo_version
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2048 + embedding_dim // 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.Tanh(),
        )
    
    def forward(self, x):
        """
        Forward pass through hybrid architecture.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        # ResNet features
        resnet_features = self.resnet(x)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        
        # YOLO features
        yolo_features = self.yolo_extractor(x)
        
        # Concatenate and fuse
        combined = torch.cat([resnet_features, yolo_features], dim=1)
        embedding = self.fusion(combined)
        
        return embedding


# Import existing models for backward compatibility
from models import ClientModel2Layers, ClientModel3Layers, ServerModel


def create_client_model(model_type='resnet_vgg', embedding_dim=64):
    """
    Factory function to create client models with YOLO support.
    
    Args:
        model_type: Model architecture type
            - 'resnet_vgg': ResNet50 + VGG19 (original)
            - 'resnet_densenet_vgg': ResNet50 + DenseNet169 + VGG19
            - 'yolo5': YOLOv5-based feature extraction
            - 'yolo8': YOLOv8-based feature extraction
            - 'resnet_yolo': Hybrid ResNet50 + YOLO
            - 'vit': Vision Transformer (if available)
            - 'vit_small': Small Vision Transformer (if available)
            - 'hybrid': Hybrid CNN + ViT (if available)
        embedding_dim: Output embedding dimension (default: 64)
    
    Returns:
        Client model instance
    
    Raises:
        ValueError: If model_type is not supported
    """
    # Original models
    if model_type == 'resnet_vgg':
        return ClientModel2Layers()
    
    elif model_type == 'resnet_densenet_vgg':
        return ClientModel3Layers()
    
    # YOLO models
    elif model_type == 'yolo5':
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        return ClientModelYOLO(embedding_dim=embedding_dim, yolo_version='yolov5')
    
    elif model_type == 'yolo8':
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        return ClientModelYOLO(embedding_dim=embedding_dim, yolo_version='yolov8')
    
    elif model_type == 'resnet_yolo':
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        return ClientModelResNetYOLO(embedding_dim=embedding_dim, yolo_version='yolov5')
    
    # Vision Transformer models (if available)
    elif model_type in ['vit', 'vit_small', 'hybrid']:
        try:
            from vit_models import ClientModelViT, ClientModelHybrid
            
            if model_type == 'vit':
                return ClientModelViT(model_name='vit_base_patch16_224', embedding_dim=embedding_dim)
            elif model_type == 'vit_small':
                return ClientModelViT(model_name='vit_small_patch16_224', embedding_dim=embedding_dim)
            elif model_type == 'hybrid':
                return ClientModelHybrid(embedding_dim=embedding_dim)
        except ImportError:
            raise ImportError(f"Vision Transformer models not available. Install timm and transformers.")
    
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: resnet_vgg, resnet_densenet_vgg, yolo5, yolo8, resnet_yolo, vit, vit_small, hybrid"
        )


def get_model_info(model):
    """
    Get detailed information about a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model statistics
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': num_params,
        'trainable_parameters': num_trainable,
        'frozen_parameters': num_params - num_trainable,
        'size_mb': num_params * 4 / (1024 ** 2),  # Assuming float32
        'model_type': model.__class__.__name__
    }


def compare_models(model_types=['resnet_vgg', 'yolo5'], embedding_dim=64):
    """
    Compare different model architectures.
    
    Args:
        model_types: List of model types to compare
        embedding_dim: Embedding dimension for all models
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*80)
    
    results = {}
    
    for model_type in model_types:
        try:
            print(f"\n{model_type.upper()}:")
            model = create_client_model(model_type, embedding_dim)
            info = get_model_info(model)
            
            print(f"  Parameters: {info['total_parameters']:,}")
            print(f"  Trainable:  {info['trainable_parameters']:,}")
            print(f"  Size:       {info['size_mb']:.2f} MB")
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224)
            start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            with torch.no_grad():
                if start:
                    start.record()
                output = model(dummy_input)
                if end:
                    end.record()
                    torch.cuda.synchronize()
                    inference_time = start.elapsed_time(end)
                else:
                    inference_time = 0
            
            print(f"  Output:     {output.shape}")
            print(f"  Inference:  {inference_time:.2f} ms" if inference_time > 0 else "  Inference:  N/A (CPU)")
            
            results[model_type] = info
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[model_type] = None
    
    print("\n" + "="*80)
    return results


if __name__ == '__main__':
    """Test YOLO model creation and comparison."""
    print("Testing YOLO Models for MedRAG VFL")
    print("="*80)
    
    # Test individual models
    if YOLO_AVAILABLE:
        print("\n1. Testing YOLOv5 model...")
        try:
            model_yolo5 = create_client_model('yolo5', embedding_dim=64)
            print("   ✓ YOLOv5 model created successfully")
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224)
            output = model_yolo5(dummy_input)
            print(f"   ✓ Forward pass successful: {output.shape}")
            assert output.shape == (2, 64), f"Expected (2, 64), got {output.shape}"
            print("   ✓ Output shape verified")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        print("\n2. Testing Hybrid ResNet+YOLO model...")
        try:
            model_hybrid = create_client_model('resnet_yolo', embedding_dim=64)
            print("   ✓ Hybrid model created successfully")
            
            # Test forward pass
            output = model_hybrid(dummy_input)
            print(f"   ✓ Forward pass successful: {output.shape}")
            assert output.shape == (2, 64), f"Expected (2, 64), got {output.shape}"
            print("   ✓ Output shape verified")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("\n⚠️  ultralytics not available. Install with: pip install ultralytics")
    
    # Compare all available models
    print("\n3. Comparing model architectures...")
    available_models = ['resnet_vgg']
    if YOLO_AVAILABLE:
        available_models.extend(['yolo5', 'resnet_yolo'])
    
    compare_models(available_models, embedding_dim=64)
    
    print("\n✓ All tests completed!")
