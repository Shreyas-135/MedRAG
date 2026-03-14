"""
Unit tests for Model Registry System

This module contains tests for the ModelRegistry class used in MedRAG for
managing and versioning trained models. Tests cover:
- Model saving and loading
- Version management
- Checkpoint storage and retrieval
- Metadata and metrics tracking
- Model comparison and best model selection
- Registry integrity and persistence

The registry provides a complete audit trail of model evolution during training.
"""

import unittest
import tempfile
import shutil
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model_registry import ModelRegistry, ModelVersion


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestModelRegistry(unittest.TestCase):
    """Test cases for ModelRegistry"""
    
    def setUp(self):
        """Create temporary directory for testing"""
        self.test_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(registry_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test registry initialization"""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertTrue((Path(self.test_dir) / 'checkpoints').exists())
        self.assertEqual(len(self.registry.versions), 0)
    
    def test_save_model(self):
        """Test model saving"""
        model = SimpleModel()
        metrics = {'accuracy': 0.85, 'loss': 0.45}
        config = {'theta': 0.1, 'use_rag': True}
        
        version_id = self.registry.save_model(
            model, round_num=1, metrics=metrics, config=config
        )
        
        self.assertIsNotNone(version_id)
        self.assertIn('v1.0_round1', version_id)
        self.assertEqual(len(self.registry.versions), 1)
        
        # Check checkpoint file exists
        version = self.registry.get_version(version_id)
        self.assertTrue(Path(version.checkpoint_path).exists())
    
    def test_load_model(self):
        """Test model loading"""
        model = SimpleModel()
        original_state = model.state_dict()
        
        version_id = self.registry.save_model(
            model, round_num=1,
            metrics={'accuracy': 0.85},
            config={'theta': 0.1}
        )
        
        # Create new model and load weights
        new_model = SimpleModel()
        loaded_model = self.registry.load_model(version_id, new_model)
        
        # Compare state dicts
        for key in original_state:
            self.assertTrue(
                torch.allclose(original_state[key], loaded_model.state_dict()[key])
            )
    
    def test_version_history(self):
        """Test getting version history"""
        model = SimpleModel()
        
        # Save multiple versions
        for i in range(3):
            self.registry.save_model(
                model, round_num=i+1,
                metrics={'accuracy': 0.7 + i*0.05},
                config={'theta': 0.1}
            )
        
        history = self.registry.get_model_history()
        self.assertEqual(len(history), 3)
        
        # Check sorted by timestamp (newest first)
        timestamps = [v.timestamp for v in history]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))
    
    def test_best_model_accuracy(self):
        """Test getting best model by accuracy"""
        model = SimpleModel()
        
        # Save models with different accuracies
        accuracies = [0.75, 0.90, 0.82]
        for i, acc in enumerate(accuracies):
            self.registry.save_model(
                model, round_num=i+1,
                metrics={'accuracy': acc, 'loss': 0.5},
                config={'theta': 0.1}
            )
        
        best = self.registry.get_best_model('accuracy')
        self.assertIsNotNone(best)
        self.assertEqual(best.metrics['accuracy'], 0.90)
    
    def test_best_model_loss(self):
        """Test getting best model by loss (lower is better)"""
        model = SimpleModel()
        
        # Save models with different losses
        losses = [0.45, 0.30, 0.52]
        for i, loss in enumerate(losses):
            self.registry.save_model(
                model, round_num=i+1,
                metrics={'accuracy': 0.8, 'loss': loss},
                config={'theta': 0.1}
            )
        
        best = self.registry.get_best_model('loss')
        self.assertIsNotNone(best)
        self.assertEqual(best.metrics['loss'], 0.30)
    
    def test_get_summary(self):
        """Test registry summary"""
        model = SimpleModel()
        
        summary = self.registry.get_summary()
        self.assertEqual(summary['total_versions'], 0)
        
        # Add a version
        self.registry.save_model(
            model, round_num=1,
            metrics={'accuracy': 0.85},
            config={'theta': 0.1}
        )
        
        summary = self.registry.get_summary()
        self.assertEqual(summary['total_versions'], 1)
        self.assertIsNotNone(summary['latest_version'])
        self.assertEqual(summary['best_accuracy'], 0.85)
    
    def test_export_registry(self):
        """Test exporting registry to JSON"""
        model = SimpleModel()
        self.registry.save_model(
            model, round_num=1,
            metrics={'accuracy': 0.85},
            config={'theta': 0.1}
        )
        
        export_path = Path(self.test_dir) / 'export.json'
        result_path = self.registry.export_registry_json(str(export_path))
        
        self.assertTrue(Path(result_path).exists())
        
        # Verify can be loaded
        import json
        with open(result_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data), 1)
    
    def test_delete_version(self):
        """Test deleting a version"""
        model = SimpleModel()
        version_id = self.registry.save_model(
            model, round_num=1,
            metrics={'accuracy': 0.85},
            config={'theta': 0.1}
        )
        
        self.assertEqual(len(self.registry.versions), 1)
        
        # Delete version
        self.registry.delete_version(version_id)
        
        self.assertEqual(len(self.registry.versions), 0)
        
        # Check checkpoint file is deleted
        version = self.registry.versions.get(version_id)
        self.assertIsNone(version)
    
    def test_model_hash_consistency(self):
        """Test that same model produces same hash"""
        model = SimpleModel()
        hash1 = self.registry._compute_model_hash(model)
        hash2 = self.registry._compute_model_hash(model)
        
        self.assertEqual(hash1, hash2)
    
    def test_model_hash_difference(self):
        """Test that different models produce different hashes"""
        model1 = SimpleModel()
        model2 = SimpleModel()
        
        # Modify model2 slightly
        with torch.no_grad():
            model2.fc.weight.fill_(0.5)
        
        hash1 = self.registry._compute_model_hash(model1)
        hash2 = self.registry._compute_model_hash(model2)
        
        self.assertNotEqual(hash1, hash2)


class TestModelVersion(unittest.TestCase):
    """Test cases for ModelVersion"""
    
    def test_to_dict(self):
        """Test converting ModelVersion to dict"""
        version = ModelVersion(
            version_id='v1.0',
            round_num=1,
            metrics={'accuracy': 0.85},
            config={'theta': 0.1},
            model_hash='abc123',
            timestamp='2023-12-23T14:30:00',
            checkpoint_path='/path/to/checkpoint.pt'
        )
        
        data = version.to_dict()
        self.assertEqual(data['version_id'], 'v1.0')
        self.assertEqual(data['round_num'], 1)
        self.assertEqual(data['metrics']['accuracy'], 0.85)
    
    def test_from_dict(self):
        """Test creating ModelVersion from dict"""
        data = {
            'version_id': 'v1.0',
            'round_num': 1,
            'metrics': {'accuracy': 0.85},
            'config': {'theta': 0.1},
            'model_hash': 'abc123',
            'timestamp': '2023-12-23T14:30:00',
            'checkpoint_path': '/path/to/checkpoint.pt'
        }
        
        version = ModelVersion.from_dict(data)
        self.assertEqual(version.version_id, 'v1.0')
        self.assertEqual(version.round_num, 1)
        self.assertEqual(version.metrics['accuracy'], 0.85)


if __name__ == '__main__':
    unittest.main()
