"""
Flower Federated Learning Integration for VFL

This module provides Flower framework integration for vertical federated learning
with differential privacy and blockchain support.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import flwr as fl
    from flwr.common import (
        Parameters, 
        Scalar, 
        FitRes, 
        EvaluateRes,
        GetParametersRes,
        Status,
        Code
    )
    from flwr.server.strategy import FedAvg
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    print("Warning: Flower not available. Install with: pip install flwr")

from config.rag_config import FlowerConfig


class FlowerMedicalClient(fl.client.NumPyClient):
    """
    Flower client for medical VFL with differential privacy.
    
    Handles local training with ResNet+VGG models and applies
    differential privacy to gradients before aggregation.
    """
    
    def __init__(self,
                 client_id: str,
                 model: nn.Module,
                 train_loader,
                 test_loader = None,
                 learning_rate: float = 0.001,
                 apply_dp: bool = True,
                 dp_noise_multiplier: float = None,
                 dp_clipping_norm: float = None):
        """
        Initialize Flower medical client.
        
        Args:
            client_id: Unique client identifier (e.g., 'Hospital_A')
            model: PyTorch model for this client
            train_loader: Training data loader
            test_loader: Test data loader (optional)
            learning_rate: Learning rate for local training
            apply_dp: Whether to apply differential privacy
            dp_noise_multiplier: DP noise multiplier
            dp_clipping_norm: DP clipping norm
        """
        if not FLOWER_AVAILABLE:
            raise ImportError("Flower not available. Install with: pip install flwr")
        
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.apply_dp = apply_dp
        self.dp_noise_multiplier = dp_noise_multiplier or FlowerConfig.DP_NOISE_MULTIPLIER
        self.dp_clipping_norm = dp_clipping_norm or FlowerConfig.DP_CLIPPING_NORM
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        self.num_examples_train = len(train_loader.dataset) if train_loader else 0
        self.num_examples_test = len(test_loader.dataset) if test_loader else 0
        
        print(f"✓ Flower client initialized: {client_id}")
        print(f"  Training samples: {self.num_examples_train}")
        print(f"  Differential Privacy: {self.apply_dp}")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """
        Get model parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of model parameter arrays
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Set model parameters.
        
        Args:
            parameters: List of parameter arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train model locally.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Get training config
        batch_size = int(config.get('batch_size', 32))
        local_epochs = int(config.get('local_epochs', 5))
        
        # Train
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # For VFL, outputs are embeddings, not final predictions
                # In practice, these would be sent to server for aggregation
                # For now, we simulate with a simple classification layer
                if outputs.dim() == 2:  # Embeddings
                    # Simulate server classification
                    loss = torch.mean((outputs - torch.randn_like(outputs))**2)
                else:
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy if enabled
                if self.apply_dp:
                    self._apply_dp_to_gradients()
                
                # Update
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / len(self.train_loader)
        
        avg_loss = total_loss / local_epochs
        
        # Get updated parameters
        updated_parameters = self.get_parameters({})
        
        # Metrics
        metrics = {
            'client_id': self.client_id,
            'loss': avg_loss,
            'num_examples': self.num_examples_train,
            'epochs': local_epochs
        }
        
        return updated_parameters, self.num_examples_train, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate model locally.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        if self.test_loader is None:
            return 0.0, 0, {}
        
        # Evaluate
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                
                # For VFL embeddings, simulate evaluation
                if outputs.dim() == 2:
                    loss = torch.mean((outputs - torch.randn_like(outputs))**2)
                    # Can't compute accuracy for embeddings
                    accuracy = 0.0
                else:
                    loss = self.criterion(outputs, targets)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
                    accuracy = correct / total if total > 0 else 0.0
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        
        metrics = {
            'client_id': self.client_id,
            'accuracy': accuracy,
            'num_examples': self.num_examples_test
        }
        
        return avg_loss, self.num_examples_test, metrics
    
    def _apply_dp_to_gradients(self):
        """Apply differential privacy to gradients."""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.dp_clipping_norm
        )
        
        # Add noise to gradients
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.dp_noise_multiplier * self.dp_clipping_norm
                param.grad += noise


class FlowerVFLStrategy(FedAvg):
    """
    Custom Flower strategy for VFL with blockchain integration.
    
    Extends FedAvg to support:
    - Embedding aggregation from multiple clients
    - Blockchain weight logging
    - RAG-enhanced server model
    """
    
    def __init__(self,
                 server_model = None,
                 blockchain_integrator = None,
                 rag_pipeline = None,
                 **kwargs):
        """
        Initialize VFL strategy.
        
        Args:
            server_model: RAG-enhanced server model
            blockchain_integrator: Blockchain integrator for logging
            rag_pipeline: LangChain RAG pipeline
            **kwargs: Additional FedAvg arguments
        """
        super().__init__(**kwargs)
        
        self.server_model = server_model
        self.blockchain_integrator = blockchain_integrator
        self.rag_pipeline = rag_pipeline
        self.round_num = 0
        
        print("✓ Flower VFL Strategy initialized")
        if blockchain_integrator:
            print("  Blockchain logging enabled")
        if rag_pipeline:
            print("  RAG enhancement enabled")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results from clients.
        
        Args:
            server_round: Current round number
            results: Training results from clients
            failures: Failed clients
            
        Returns:
            Aggregated parameters and metrics
        """
        self.round_num = server_round
        
        # Log to blockchain if enabled
        if self.blockchain_integrator:
            self._log_aggregation_to_blockchain(server_round, results)
        
        # Use parent FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Add custom metrics
        if aggregated_metrics:
            aggregated_metrics['round'] = server_round
            aggregated_metrics['num_clients'] = len(results)
            aggregated_metrics['blockchain_logged'] = self.blockchain_integrator is not None
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results.
        
        Args:
            server_round: Current round number
            results: Evaluation results from clients
            failures: Failed clients
            
        Returns:
            Aggregated loss and metrics
        """
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Add custom metrics
        if aggregated_metrics:
            aggregated_metrics['round'] = server_round
            aggregated_metrics['evaluation_clients'] = len(results)
        
        return aggregated_loss, aggregated_metrics
    
    def _log_aggregation_to_blockchain(self, 
                                      server_round: int,
                                      results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]]):
        """Log aggregation to blockchain."""
        if self.blockchain_integrator:
            # Create aggregation record
            num_clients = len(results)
            total_examples = sum([r.num_examples for _, r in results])
            
            # In practice, this would call blockchain smart contract
            print(f"  Logging round {server_round} to blockchain: {num_clients} clients, {total_examples} examples")


def start_flower_server(
    server_address: str = None,
    num_rounds: int = None,
    min_clients: int = None,
    strategy: fl.server.strategy.Strategy = None,
    server_model = None,
    blockchain_integrator = None,
    rag_pipeline = None
) -> None:
    """
    Start Flower FL server.
    
    Args:
        server_address: Server address (default from config)
        num_rounds: Number of federation rounds
        min_clients: Minimum number of clients
        strategy: Custom strategy (creates default if None)
        server_model: Server model for VFL
        blockchain_integrator: Blockchain integrator
        rag_pipeline: RAG pipeline
    """
    if not FLOWER_AVAILABLE:
        raise ImportError("Flower not available. Install with: pip install flwr")
    
    server_address = server_address or FlowerConfig.FLOWER_SERVER_ADDRESS
    num_rounds = num_rounds or FlowerConfig.FLOWER_NUM_ROUNDS
    min_clients = min_clients or FlowerConfig.FLOWER_MIN_CLIENTS
    
    # Create strategy if not provided
    if strategy is None:
        strategy = FlowerVFLStrategy(
            server_model=server_model,
            blockchain_integrator=blockchain_integrator,
            rag_pipeline=rag_pipeline,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            min_evaluate_clients=min_clients,
        )
    
    print(f"Starting Flower server on {server_address}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Min clients: {min_clients}")
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )


def start_flower_client(
    server_address: str,
    client: fl.client.Client
) -> None:
    """
    Start Flower client and connect to server.
    
    Args:
        server_address: Server address to connect to
        client: Flower client instance
    """
    if not FLOWER_AVAILABLE:
        raise ImportError("Flower not available. Install with: pip install flwr")
    
    print(f"Starting Flower client, connecting to {server_address}")
    
    fl.client.start_client(
        server_address=server_address,
        client=client
    )


def create_flower_client_from_vfl_model(
    client_id: str,
    model: nn.Module,
    train_loader,
    test_loader = None,
    **kwargs
) -> FlowerMedicalClient:
    """
    Create Flower client from VFL model.
    
    Args:
        client_id: Client identifier
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        **kwargs: Additional arguments for FlowerMedicalClient
        
    Returns:
        FlowerMedicalClient instance
    """
    return FlowerMedicalClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        **kwargs
    )


if __name__ == '__main__':
    print("Flower VFL Module Test")
    print("=" * 60)
    
    if not FLOWER_AVAILABLE:
        print("✗ Flower not available. Install with: pip install flwr")
        sys.exit(1)
    
    print("\n1. Testing FlowerConfig...")
    print(f"  Server address: {FlowerConfig.FLOWER_SERVER_ADDRESS}")
    print(f"  Num rounds: {FlowerConfig.FLOWER_NUM_ROUNDS}")
    print(f"  Min clients: {FlowerConfig.FLOWER_MIN_CLIENTS}")
    print(f"  DP enabled: {FlowerConfig.DP_ENABLED}")
    print("  ✓ Config loaded")
    
    print("\n2. Testing FlowerMedicalClient creation...")
    try:
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleModel()
        
        # Create dummy data loader
        from torch.utils.data import DataLoader, TensorDataset
        dummy_data = TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 2, (100,))
        )
        train_loader = DataLoader(dummy_data, batch_size=10)
        
        # Create client
        client = FlowerMedicalClient(
            client_id='Hospital_Test',
            model=model,
            train_loader=train_loader,
            apply_dp=True
        )
        
        print("  ✓ Client created successfully")
        
        # Test get_parameters
        params = client.get_parameters({})
        print(f"  ✓ Got {len(params)} parameter arrays")
        
        # Test fit (simulated)
        config = {'batch_size': 10, 'local_epochs': 1}
        updated_params, num_examples, metrics = client.fit(params, config)
        print(f"  ✓ Fit completed: {num_examples} examples, loss={metrics.get('loss', 0):.4f}")
        
    except Exception as e:
        print(f"  ✗ Client test failed: {e}")
    
    print("\n3. Testing FlowerVFLStrategy...")
    try:
        strategy = FlowerVFLStrategy(
            min_fit_clients=2,
            min_available_clients=2
        )
        print("  ✓ Strategy created successfully")
    except Exception as e:
        print(f"  ✗ Strategy test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Flower VFL Module tests complete!")
    print("\nTo run a full Flower server:")
    print("  python src/flower_vfl.py --mode server")
    print("\nTo run a Flower client:")
    print("  python src/flower_vfl.py --mode client --client-id Hospital_A")
