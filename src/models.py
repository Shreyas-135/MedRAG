import torch
import torch.nn as nn
import torchvision

class ClientModel2Layers(nn.Module):
    def __init__(self):
        super(ClientModel2Layers, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.vgg = torchvision.models.vgg19(pretrained=True)

        # Remove the classification layers (fully connected layers)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(27136, 64),
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.resnet(x)
        x2 = self.vgg(x)

        # Flatten and concatenate
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)

        # Final embedding has size 64 and in range [-1, 1]
        x = self.classifier(x)

        return x
    
class ClientModel3Layers(nn.Module):
    def __init__(self):
        super(ClientModel3Layers, self).__init__()
        self.densenet = torchvision.models.densenet169(pretrained=True)
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.vgg = torchvision.models.vgg19(pretrained=True)

        # Remove the classification layers (fully connected layers)
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(108672, 64),
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.resnet(x)
        x3 = self.vgg(x)

        # Flatten and concatenate
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x = torch.cat((x1, x2, x3), dim=1)

        # Final embedding has size 64 and in range [-1, 1]
        x = self.classifier(x)

        return x

class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()

        # classification layers
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.BatchNorm1d(64),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# ============================================================================
# Model Factory Function (Added for 2025 Upgrade)
# ============================================================================

def create_client_model(model_type='resnet_vgg', embedding_dim=64):
    """
    Factory function to create a client model.

    Supported types (legacy two-class VFL clients kept for backward
    compatibility with ``inference.py`` fallback path):
        - ``'resnet_vgg'``: ResNet50 + VGG19 dual-stream (default)

    For the main training pipeline the :class:`~vfl_feature_partition.VFLFramework`
    class is used instead (backbones: resnet18, densenet121, efficientnet_b0).

    Args:
        model_type  : model architecture string (only 'resnet_vgg' is active)
        embedding_dim: output embedding dimension (used by newer architectures)

    Returns:
        Client model instance
    """
    if model_type == 'resnet_vgg':
        return ClientModel2Layers()

    raise ValueError(
        f"Unknown model type: {model_type!r}. "
        f"For the main pipeline use VFLFramework (resnet18, densenet121, efficientnet_b0)."
    )


def get_model_info(model):
    """
    Get information about a model
    
    Returns:
        Dictionary with model stats
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': num_params,
        'trainable_parameters': num_trainable,
        'size_mb': num_params * 4 / (1024 ** 2),  # Assuming float32
        'model_type': model.__class__.__name__
    }
