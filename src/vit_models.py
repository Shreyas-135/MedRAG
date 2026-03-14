"""
Vision Transformer Models for Medical Image Feature Extraction
Supports ViT-Base and ViT-Small for federated learning clients
"""

import torch
import torch.nn as nn
import timm

class ClientModelViT(nn.Module):
    """Vision Transformer client model for federated learning"""
    
    def __init__(self, model_name='vit_base_patch16_224', embedding_dim=64):
        super(ClientModelViT, self).__init__()
        
        # Load pre-trained ViT from timm
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Get ViT output dimension
        if 'base' in model_name:
            vit_dim = 768
        elif 'small' in model_name:
            vit_dim = 384
        elif 'large' in model_name:
            vit_dim = 1024
        else: 
            vit_dim = 768
        
        # Feature projection to match VFL embedding dimension
        self.classifier = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.Tanh(),
        )
    
    def forward(self, x):
        # Extract features with ViT
        features = self.vit(x)
        
        # Project to embedding space
        embedding = self.classifier(features)
        return embedding


class ClientModelHybrid(nn.Module):
    """Hybrid CNN + ViT client model combining ResNet and ViT"""
    
    def __init__(self, embedding_dim=64):
        super(ClientModelHybrid, self).__init__()
        
        # ResNet branch
        import torchvision
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # ViT branch (smaller for efficiency)
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 384, 512),  # ResNet50 (2048) + ViT-Small (384)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.Tanh(),
        )
    
    def forward(self, x):
        # ResNet features
        resnet_features = self.resnet(x)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        
        # ViT features
        vit_features = self.vit(x)
        
        # Concatenate and fuse
        combined = torch.cat((resnet_features, vit_features), dim=1)
        embedding = self.fusion(combined)
        
        return embedding
