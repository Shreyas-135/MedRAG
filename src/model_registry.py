"""
Model Registry and Versioning System for MedRAG
Tracks model versions, checkpoints, metrics, and training configurations.
"""

import os
import json
import torch
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import shutil


class ModelVersion:
    """Represents a single model version with metadata."""
    
    def __init__(self, version_id: str, round_num: int, metrics: Dict[str, float],
                 config: Dict[str, Any], model_hash: str, timestamp: str,
                 checkpoint_path: str):
        self.version_id = version_id
        self.round_num = round_num
        self.metrics = metrics
        self.config = config
        self.model_hash = model_hash
        self.timestamp = timestamp
        self.checkpoint_path = checkpoint_path
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            'version_id': self.version_id,
            'round_num': self.round_num,
            'metrics': self.metrics,
            'config': self.config,
            'model_hash': self.model_hash,
            'sha256': self.model_hash,
            'timestamp': self.timestamp,
            'checkpoint_path': self.checkpoint_path,
        }
        # Promote backbone / hospital to top level for registry readability
        if isinstance(self.config, dict):
            backbone = self.config.get('backbone_name') or self.config.get('backbone')
            if backbone:
                d['backbone'] = backbone
                d['model_name'] = backbone
            hospital = self.config.get('hospital')
            if hospital:
                d['hospital'] = hospital
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """Create ModelVersion from dictionary."""
        return cls(
            version_id=data['version_id'],
            round_num=data['round_num'],
            metrics=data['metrics'],
            config=data['config'],
            model_hash=data['model_hash'],
            timestamp=data['timestamp'],
            checkpoint_path=data['checkpoint_path']
        )


class ModelRegistry:
    """
    Model registry for tracking and managing model versions.
    
    Features:
    - Save/load models with version tracking
    - Track performance metrics per version
    - Store training configuration
    - Calculate model checksums (SHA-256)
    - Export registry to JSON
    """
    
    def __init__(self, registry_dir: str = None):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store registry and checkpoints.
                         Defaults to 'models/registry' relative to project root.
        """
        if registry_dir is None:
            # Default to models/registry in project root
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            registry_dir = os.path.join(repo_root, 'models', 'registry')
        
        self.registry_dir = Path(registry_dir)
        self.checkpoint_dir = self.registry_dir / 'checkpoints'
        self.registry_file = self.registry_dir / 'registry.json'
        
        # Create directories if they don't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry or create new
        self.versions: Dict[str, ModelVersion] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from JSON file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.versions = {
                        vid: ModelVersion.from_dict(vdata) 
                        for vid, vdata in data.items()
                    }
            except Exception as e:
                print(f"Warning: Could not load registry: {e}")
                self.versions = {}
        else:
            self.versions = {}
    
    def _save_registry(self):
        """Save registry to JSON file."""
        data = {vid: v.to_dict() for vid, v in self.versions.items()}
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register_entry(self, version: 'ModelVersion') -> str:
        """
        Register a pre-built :class:`ModelVersion` entry (e.g. for an
        already-saved checkpoint) and persist the registry.

        This is the public counterpart of the internal ``_save_registry``
        helper and avoids callers having to access private methods.

        Args:
            version: Fully populated :class:`ModelVersion` instance.

        Returns:
            The ``version_id`` of the registered entry.
        """
        self.versions[version.version_id] = version
        self._save_registry()
        return version.version_id
    
    def _compute_model_hash(self, model: torch.nn.Module) -> str:
        """
        Compute SHA-256 hash of model weights.
        
        Args:
            model: PyTorch model
            
        Returns:
            Hexadecimal hash string
        """
        # Serialize model state dict to bytes for consistent hashing
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        model_bytes = buffer.getvalue()
        return hashlib.sha256(model_bytes).hexdigest()
    
    def _generate_version_id(self, round_num: int) -> str:
        """Generate version ID based on round number and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Format: v1.0_round5_20231223_143022
        major = 1
        minor = len(self.versions)
        return f"v{major}.{minor}_round{round_num}_{timestamp}"
    
    def save_model(self, model: torch.nn.Module, round_num: int, 
                   metrics: Dict[str, float], config: Dict[str, Any]) -> str:
        """
        Save model checkpoint with metadata.
        
        Args:
            model: PyTorch model to save
            round_num: Training round number
            metrics: Performance metrics (e.g., {'accuracy': 0.85, 'loss': 0.45})
            config: Training configuration (theta, num_clients, use_rag, etc.)
            
        Returns:
            version_id: Unique version identifier
        """
        # Generate version ID
        version_id = self._generate_version_id(round_num)
        
        # Compute model hash
        model_hash = self._compute_model_hash(model)
        
        # Save checkpoint
        checkpoint_filename = f"{version_id}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'round_num': round_num,
            'metrics': metrics,
            'config': config,
            'model_hash': model_hash,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        # Create version entry
        version = ModelVersion(
            version_id=version_id,
            round_num=round_num,
            metrics=metrics,
            config=config,
            model_hash=model_hash,
            timestamp=datetime.now().isoformat(),
            checkpoint_path=str(checkpoint_path)
        )
        
        # Add to registry
        self.versions[version_id] = version
        self._save_registry()
        
        return version_id
    
    def load_model(self, version_id: str, model: torch.nn.Module = None) -> torch.nn.Module:
        """
        Load model from checkpoint.
        
        Args:
            version_id: Version identifier
            model: Optional model instance to load weights into.
                  If None, returns state dict.
            
        Returns:
            Loaded model or state dict
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found in registry")
        
        version = self.versions[version_id]
        checkpoint_path = Path(version.checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        else:
            return checkpoint['model_state_dict']
    
    def get_model_history(self) -> List[ModelVersion]:
        """
        Get all model versions sorted by timestamp.
        
        Returns:
            List of ModelVersion objects
        """
        return sorted(
            self.versions.values(), 
            key=lambda v: v.timestamp,
            reverse=True
        )
    
    def get_best_model(self, metric: str = 'accuracy') -> Optional[ModelVersion]:
        """
        Get the best model based on a specific metric.
        
        Args:
            metric: Metric to optimize (e.g., 'accuracy', 'loss')
            
        Returns:
            ModelVersion with best metric, or None if no versions exist
        """
        if not self.versions:
            return None
        
        # For loss, lower is better; for accuracy, higher is better
        reverse = metric != 'loss'
        
        versions_with_metric = [
            v for v in self.versions.values() 
            if metric in v.metrics
        ]
        
        if not versions_with_metric:
            return None
        
        return max(
            versions_with_metric,
            key=lambda v: v.metrics[metric] if reverse else -v.metrics[metric]
        )
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get specific version by ID."""
        return self.versions.get(version_id)
    
    def export_registry_json(self, output_path: str = None) -> str:
        """
        Export registry to JSON file.
        
        Args:
            output_path: Output file path. If None, uses default registry file.
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = self.registry_file
        else:
            output_path = Path(output_path)
        
        data = {vid: v.to_dict() for vid, v in self.versions.items()}
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(output_path)
    
    def delete_version(self, version_id: str):
        """
        Delete a model version and its checkpoint.
        
        Args:
            version_id: Version to delete
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        checkpoint_path = Path(version.checkpoint_path)
        
        # Delete checkpoint file
        if checkpoint_path.exists():
            os.remove(checkpoint_path)
        
        # Remove from registry
        del self.versions[version_id]
        self._save_registry()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the registry."""
        if not self.versions:
            return {
                'total_versions': 0,
                'latest_version': None,
                'best_accuracy': None,
                'storage_size_mb': 0
            }
        
        # Calculate total storage size
        total_size = sum(
            os.path.getsize(Path(v.checkpoint_path)) 
            for v in self.versions.values()
            if Path(v.checkpoint_path).exists()
        )
        
        latest = max(self.versions.values(), key=lambda v: v.timestamp)
        best = self.get_best_model('accuracy')
        
        return {
            'total_versions': len(self.versions),
            'latest_version': latest.version_id,
            'best_accuracy': best.metrics.get('accuracy') if best else None,
            'storage_size_mb': total_size / (1024 * 1024)
        }


if __name__ == "__main__":
    # Example usage
    print("Model Registry Demo")
    print("=" * 50)
    
    # Initialize registry
    registry = ModelRegistry()
    print(f"✓ Registry initialized at: {registry.registry_dir}")
    
    # Get summary
    summary = registry.get_summary()
    print(f"\nRegistry Summary:")
    print(f"  Total versions: {summary['total_versions']}")
    print(f"  Storage size: {summary['storage_size_mb']:.2f} MB")
    
    if summary['total_versions'] > 0:
        print(f"  Latest version: {summary['latest_version']}")
        if summary['best_accuracy']:
            print(f"  Best accuracy: {summary['best_accuracy']:.2%}")
