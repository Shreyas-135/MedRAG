"""
Tests for Flower VFL Components

This module tests:
- Flower client initialization
- Server strategy
- Gradient aggregation
- Integration with blockchain
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestFlowerClient(unittest.TestCase):
    """Test Flower medical client."""
    
    def test_client_initialization(self):
        """Test client initialization."""
        try:
            from flower_vfl import FlowerMedicalClient
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            # Create simple model
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 64)
                
                def forward(self, x):
                    return self.fc(x)
            
            model = SimpleModel()
            
            # Create dummy data
            dummy_data = TensorDataset(
                torch.randn(50, 10),
                torch.randint(0, 2, (50,))
            )
            train_loader = DataLoader(dummy_data, batch_size=10)
            
            # Create client
            client = FlowerMedicalClient(
                client_id='Hospital_Test',
                model=model,
                train_loader=train_loader,
                apply_dp=True
            )
            
            self.assertEqual(client.client_id, 'Hospital_Test')
            self.assertEqual(client.num_examples_train, 50)
            self.assertTrue(client.apply_dp)
            
            print("  ✓ Client initialization test passed")
            
        except ImportError:
            self.skipTest("Flower not available")
    
    def test_get_parameters(self):
        """Test getting model parameters."""
        try:
            from flower_vfl import FlowerMedicalClient
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)
                
                def forward(self, x):
                    return self.fc(x)
            
            model = SimpleModel()
            dummy_data = TensorDataset(
                torch.randn(50, 10),
                torch.randint(0, 2, (50,))
            )
            train_loader = DataLoader(dummy_data, batch_size=10)
            
            client = FlowerMedicalClient(
                client_id='Hospital_Test',
                model=model,
                train_loader=train_loader
            )
            
            # Get parameters
            params = client.get_parameters({})
            
            self.assertIsInstance(params, list)
            self.assertGreater(len(params), 0)
            self.assertIsInstance(params[0], np.ndarray)
            
            print("  ✓ Get parameters test passed")
            
        except ImportError:
            self.skipTest("Flower not available")
    
    def test_set_parameters(self):
        """Test setting model parameters."""
        try:
            from flower_vfl import FlowerMedicalClient
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)
                
                def forward(self, x):
                    return self.fc(x)
            
            model = SimpleModel()
            dummy_data = TensorDataset(
                torch.randn(50, 10),
                torch.randint(0, 2, (50,))
            )
            train_loader = DataLoader(dummy_data, batch_size=10)
            
            client = FlowerMedicalClient(
                client_id='Hospital_Test',
                model=model,
                train_loader=train_loader
            )
            
            # Get original parameters
            original_params = client.get_parameters({})
            
            # Create new parameters
            new_params = [np.random.randn(*p.shape) for p in original_params]
            
            # Set new parameters
            client.set_parameters(new_params)
            
            # Verify they were set
            updated_params = client.get_parameters({})
            
            for new, updated in zip(new_params, updated_params):
                self.assertTrue(np.allclose(new, updated))
            
            print("  ✓ Set parameters test passed")
            
        except ImportError:
            self.skipTest("Flower not available")
    
    def test_fit(self):
        """Test local training (fit)."""
        try:
            from flower_vfl import FlowerMedicalClient
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 64)
                
                def forward(self, x):
                    return self.fc(x)
            
            model = SimpleModel()
            dummy_data = TensorDataset(
                torch.randn(50, 10),
                torch.randint(0, 2, (50,))
            )
            train_loader = DataLoader(dummy_data, batch_size=10)
            
            client = FlowerMedicalClient(
                client_id='Hospital_Test',
                model=model,
                train_loader=train_loader,
                apply_dp=False  # Disable DP for test simplicity
            )
            
            # Get initial parameters
            initial_params = client.get_parameters({})
            
            # Perform training
            config = {'batch_size': 10, 'local_epochs': 1}
            updated_params, num_examples, metrics = client.fit(initial_params, config)
            
            self.assertEqual(num_examples, 50)
            self.assertIn('loss', metrics)
            self.assertIn('client_id', metrics)
            self.assertIsInstance(updated_params, list)
            
            print("  ✓ Fit test passed")
            
        except ImportError:
            self.skipTest("Flower not available")
    
    def test_differential_privacy(self):
        """Test differential privacy application."""
        try:
            from flower_vfl import FlowerMedicalClient
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 64)
                
                def forward(self, x):
                    return self.fc(x)
            
            model = SimpleModel()
            dummy_data = TensorDataset(
                torch.randn(50, 10),
                torch.randint(0, 2, (50,))
            )
            train_loader = DataLoader(dummy_data, batch_size=10)
            
            # Create client with DP
            client_with_dp = FlowerMedicalClient(
                client_id='Hospital_DP',
                model=model,
                train_loader=train_loader,
                apply_dp=True,
                dp_noise_multiplier=0.1
            )
            
            # Verify DP settings
            self.assertTrue(client_with_dp.apply_dp)
            self.assertEqual(client_with_dp.dp_noise_multiplier, 0.1)
            
            print("  ✓ Differential privacy test passed")
            
        except ImportError:
            self.skipTest("Flower not available")


class TestFlowerStrategy(unittest.TestCase):
    """Test Flower VFL strategy."""
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        try:
            from flower_vfl import FlowerVFLStrategy
            
            strategy = FlowerVFLStrategy(
                min_fit_clients=2,
                min_available_clients=2,
                min_evaluate_clients=2
            )
            
            self.assertIsNotNone(strategy)
            self.assertEqual(strategy.round_num, 0)
            
            print("  ✓ Strategy initialization test passed")
            
        except ImportError:
            self.skipTest("Flower not available")
    
    def test_strategy_with_blockchain(self):
        """Test strategy with blockchain integration."""
        try:
            from flower_vfl import FlowerVFLStrategy
            
            # Mock blockchain integrator
            mock_blockchain = Mock()
            
            strategy = FlowerVFLStrategy(
                blockchain_integrator=mock_blockchain,
                min_fit_clients=2,
                min_available_clients=2
            )
            
            self.assertIsNotNone(strategy.blockchain_integrator)
            
            print("  ✓ Strategy with blockchain test passed")
            
        except ImportError:
            self.skipTest("Flower not available")
    
    def test_strategy_with_rag(self):
        """Test strategy with RAG pipeline."""
        try:
            from flower_vfl import FlowerVFLStrategy
            
            # Mock RAG pipeline
            mock_rag = Mock()
            
            strategy = FlowerVFLStrategy(
                rag_pipeline=mock_rag,
                min_fit_clients=2,
                min_available_clients=2
            )
            
            self.assertIsNotNone(strategy.rag_pipeline)
            
            print("  ✓ Strategy with RAG test passed")
            
        except ImportError:
            self.skipTest("Flower not available")


class TestFlowerIntegration(unittest.TestCase):
    """Test Flower server/client integration."""
    
    def test_client_creation_helper(self):
        """Test client creation helper function."""
        try:
            from flower_vfl import create_flower_client_from_vfl_model
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 64)
                
                def forward(self, x):
                    return self.fc(x)
            
            model = SimpleModel()
            dummy_data = TensorDataset(
                torch.randn(50, 10),
                torch.randint(0, 2, (50,))
            )
            train_loader = DataLoader(dummy_data, batch_size=10)
            
            # Create client using helper
            client = create_flower_client_from_vfl_model(
                client_id='Hospital_Helper',
                model=model,
                train_loader=train_loader
            )
            
            self.assertEqual(client.client_id, 'Hospital_Helper')
            
            print("  ✓ Client creation helper test passed")
            
        except ImportError:
            self.skipTest("Flower not available")


def run_tests():
    """Run all tests."""
    print("Running Flower VFL Tests")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFlowerClient))
    suite.addTests(loader.loadTestsFromTestCase(TestFlowerStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestFlowerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
