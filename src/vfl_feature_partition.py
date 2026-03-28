"""
VFL Feature Partition Module

Implements vertical feature partitioning for the VFL framework.
In true VFL, different hospitals own different feature columns for the same patients.
Here we simulate VFL by splitting CNN embedding vectors into N partitions,
where each partition belongs to one hospital's feature space.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class FeaturePartitioner:
    """
    Splits and re-assembles embedding vectors for simulated VFL.

    In the simulation:
    - A CNN backbone produces a full embedding E (e.g., 512-d)
    - The embedding is split into num_partitions chunks
    - Each chunk represents the feature contribution of one hospital/party
    - The server aggregates all chunks to make the final prediction
    """

    def __init__(self, embedding_dim: int = 512, num_partitions: int = 4):
        assert embedding_dim % num_partitions == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by "
            f"num_partitions ({num_partitions})"
        )
        self.embedding_dim = embedding_dim
        self.num_partitions = num_partitions
        self.partition_dim = embedding_dim // num_partitions

    def split(self, embedding: torch.Tensor) -> List[torch.Tensor]:
        """Split embedding into num_partitions equal chunks."""
        return torch.split(embedding, self.partition_dim, dim=-1)

    def aggregate(self, partitions: List[torch.Tensor]) -> torch.Tensor:
        """Re-assemble partitions into full embedding via concatenation."""
        return torch.cat(partitions, dim=-1)

    def partition_summary(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "num_partitions": self.num_partitions,
            "partition_dim": self.partition_dim,
        }


class VFLBottomModel(nn.Module):
    """
    Hospital-side bottom model: CNN backbone that produces a feature embedding.
    The embedding is later split and only the hospital's partition is retained locally.
    """

    def __init__(self, backbone_name: str = "resnet18", embedding_dim: int = 512):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.backbone, self.feat_dim = self._build_backbone(backbone_name)
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def _build_backbone(self, name: str) -> Tuple[nn.Module, int]:
        import torchvision.models as M

        if name == "resnet18":
            m = M.resnet18(weights=M.ResNet18_Weights.DEFAULT)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
            return m, feat_dim
        elif name == "densenet121":
            m = M.densenet121(weights=M.DenseNet121_Weights.DEFAULT)
            feat_dim = m.classifier.in_features
            m.classifier = nn.Identity()
            return m, feat_dim
        elif name == "efficientnet_b0":
            m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.DEFAULT)
            feat_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m, feat_dim
        elif name == "mobilenet_v2":
            m = M.mobilenet_v2(weights=M.MobileNet_V2_Weights.DEFAULT)
            feat_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m, feat_dim
        else:
            raise ValueError(f"Unknown backbone: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        return self.proj(feats)


class VFLTopModel(nn.Module):
    """
    Server-side top model: takes concatenated feature partitions from all hospitals
    and produces the final class prediction.
    """

    def __init__(
        self,
        total_embedding_dim: int = 512,
        num_classes: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class VFLFramework(nn.Module):
    """
    End-to-end VFL framework combining bottom and top models.

    In training/demo mode, all hospitals run on the same machine.
    The forward pass simulates the VFL protocol:
      1. Each hospital's data goes through the shared bottom model (or each
         hospital could have its own bottom model in a more complex setup).
      2. The embedding is split into partitions (one per hospital).
      3. In a real VFL setup, each hospital would keep only its partition and
         send it (encrypted) to the server.
      4. The server concatenates partitions and runs the top model.

    Here we use a single shared bottom model for training convenience,
    but the feature-split logic mirrors a real VFL deployment.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        embedding_dim: int = 512,
        num_partitions: int = 4,
        num_classes: int = 4,
        top_hidden: int = 256,
    ):
        super().__init__()
        self.partitioner = FeaturePartitioner(embedding_dim, num_partitions)
        self.bottom = VFLBottomModel(backbone_name, embedding_dim)
        self.top = VFLTopModel(embedding_dim, num_classes, top_hidden)
        self.backbone_name = backbone_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.bottom(x)
        # VFL split: partition embedding (simulates vertical feature split)
        partitions = self.partitioner.split(embedding)
        # Aggregate partitions (simulates server receiving all partitions)
        aggregated = self.partitioner.aggregate(list(partitions))
        return self.top(aggregated)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get full embedding for analysis."""
        with torch.no_grad():
            return self.bottom(x)
