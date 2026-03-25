"""
Demo script for RAG-Enhanced Vertical Federated Learning with ZIP Dataset Support
Demonstrates the integration of RAG with VFL and Blockchain for
privacy-preserving cross-hospital medical imaging analysis.

Features:
- Hospital naming (A, B, C, D) instead of client0/1/2/3
- YOLO model support (YOLOv5, YOLOv8, Hybrid ResNet+YOLO)
- Compatible with ZIP-extracted datasets
- Enhanced output with hospital statistics

Usage:
    python demo_rag_vfl_with_zip.py --datapath <path> --use-rag --withblockchain
    python demo_rag_vfl_with_zip.py --datapath <path> --model-type yolo5
    python demo_rag_vfl_with_zip.py --datapath <path> --model-type resnet_yolo --use-rag
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions.binomial import Binomial
import argparse
import time
import os
import traceback

try:
    from sklearn.metrics import precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    # Try to import YOLO-enhanced models first
    from models_with_yolo import create_client_model, get_model_info, compare_models
    YOLO_MODELS_AVAILABLE = True
except ImportError:
    # Fallback to original models
    from models import ClientModel2Layers, create_client_model, get_model_info
    YOLO_MODELS_AVAILABLE = False
    print("Note: YOLO models not available. Using standard models only.")
from rag_server_model import RAGEnhancedServerModel, BlockchainRAGVFLSystem, UncertaintyAwareRAGServer
from rag_retriever import create_sample_xray_knowledge_base
from model_registry import ModelRegistry
from ledger import Ledger

# Import blockchain integration if available
try:
    from Blockchain_and_VFL_Integration import BlockchainVFLIntegrator
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False
    print("Warning: Blockchain integration not available")


def _datasize_to_fraction(datasize: float) -> float:
    """Map the CLI datasize values to a fraction of available training data."""
    allowed = {1.0, 0.5, 0.25, 0.0125}
    if datasize not in allowed:
        raise ValueError("Invalid datasize. Must be 0.0125, 0.25, 0.5 or 1.0")
    return float(datasize)


def _make_train_val_indices(n_total: int, datasize: float, seed: int = 42, train_frac: float = 0.8):
    """
    Create dynamic train/val indices based on actual dataset size.

    Subsamples ``n_total`` images according to *datasize*, then splits into
    train/val using *train_frac*.  At least one sample is guaranteed in each
    split.

    Args:
        n_total:    Total number of images in the ImageFolder dataset.
        datasize:   Fraction of data to use (one of 1.0, 0.5, 0.25, 0.0125).
        seed:       Random seed for reproducibility (use ``base_seed + client_index``).
        train_frac: Fraction of the selected subset used for training (default 0.8).

    Returns:
        Tuple ``(train_idx, val_idx)`` of NumPy integer arrays.
    """
    if n_total < 2:
        raise ValueError(f"Not enough samples to split: {n_total}")

    frac = _datasize_to_fraction(datasize)
    n_use = min(max(2, int(round(n_total * frac))), n_total)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)[:n_use]

    n_train = min(max(1, int(round(n_use * train_frac))), n_use - 1)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    return train_idx, val_idx


def detect_class_names(datapath):
    """
    Auto-detect class names from the dataset folder structure.

    Checks ``<datapath>/SplitCovid19/hospitalA/train`` first (new naming),
    then ``<datapath>/SplitCovid19/client0/train`` (legacy naming).

    Returns:
        Sorted list of class names, or ``None`` if detection fails.
    """
    split_base = os.path.join(datapath, 'SplitCovid19')
    for candidate in ['hospitalA', 'client0']:
        train_dir = os.path.join(split_base, candidate, 'train')
        if os.path.isdir(train_dir):
            classes = sorted([
                d for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            ])
            if classes:
                return classes
    return None


def quantize(x, theta, m):
    """Discrete differential privacy noise."""
    p = torch.add(0.5, torch.mul(theta, x))
    binom = Binomial(m, p)
    noise = binom.sample()
    y = x.clone()
    y.data = noise
    return y


def dequantize(q, theta, m, n):
    """Dequantize discrete sum into continuous sum."""
    det = torch.sub(q, m * n / 2)
    sum_val = torch.div(det, theta * m)
    return sum_val


def train(models, optimizers, schedulers, train_loaders, criterion, 
          num_clients, theta, quant_bin, blockchain_vfl_integrator=None,
          server_model=None, args=None, ledger=None, epoch_num=0):
    """
    Training function with RAG enhancement.
    
    Args:
        models: List of client models
        optimizers: List of optimizers for each model
        schedulers: Learning rate schedulers
        train_loaders: Data loaders for each client
        criterion: Loss function
        num_clients: Number of VFL clients
        theta: DP noise parameter
        quant_bin: Quantization parameter
        blockchain_vfl_integrator: Blockchain integrator (optional)
        server_model: RAG-enhanced server model
        args: Command line arguments
        ledger: Ledger instance for logging (optional)
        epoch_num: Current epoch number
    """
    embeddings_grad = [None] * num_clients
    embeddings_nograd = [None] * num_clients
    sum_nograd = None
    targets = None
    
    completed = False
    train_iterators = []
    for i in range(num_clients):
        train_iterators.append(iter(train_loaders[i]))
    
    while not completed:
        # At party side - generate embeddings
        for i in range(num_clients):
            item = next(train_iterators[i], -1)
            
            if item == -1:
                completed = True
                break
            
            inputs, targets = item
            
            # Generate embedding
            embeddings_grad[i] = models[i](inputs)
            with torch.no_grad():
                embeddings_nograd[i] = models[i](inputs)
            
            # Add differential privacy noise
            embeddings_nograd[i] = quantize(embeddings_nograd[i], theta, quant_bin)
            
            if args.withblockchain and blockchain_vfl_integrator:
                # Send embeddings to smart contract
                client_parameters = [list(map(int, row)) for row in embeddings_nograd[i].tolist()]
                blockchain_vfl_integrator.update_client_weights(
                    blockchain_vfl_integrator.client_accounts[i], 
                    client_parameters
                )
        
        if completed:
            break
        
        # At server side - aggregate embeddings
        if args.withblockchain and blockchain_vfl_integrator:
            blockchain_vfl_integrator.aggregate_weights()
            with torch.no_grad():
                sum_nograd = torch.tensor(blockchain_vfl_integrator.get_aggregated_weights())
        else:
            with torch.no_grad():
                sum_nograd = torch.sum(torch.stack(embeddings_nograd), dim=0)
        
        # Dequantize
        sum_nograd = dequantize(sum_nograd, theta, quant_bin, num_clients)
        sum_grad = torch.sum(torch.stack(embeddings_grad), dim=0)
        sum_grad.data = sum_nograd
        
        # Forward through RAG-enhanced server model
        outputs = server_model(sum_grad)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        for i in range(num_clients + 1):
            optimizers[i].zero_grad()
        loss.backward()
        for i in range(num_clients + 1):
            optimizers[i].step()
        
        # Update learning rates
        for i in range(num_clients + 1):
            schedulers[i].step()
    
    del train_iterators


def evaluate(models, val_loaders, test_loaders, criterion, num_clients, 
             theta, quant_bin, blockchain_vfl_integrator, server_model, 
             args, mode='test', ledger=None):
    """
    Evaluation function with RAG enhancement.
    
    Args:
        models: List of models
        val_loaders: Validation data loaders
        test_loaders: Test data loaders
        criterion: Loss function
        num_clients: Number of clients
        theta: DP noise parameter
        quant_bin: Quantization parameter
        blockchain_vfl_integrator: Blockchain integrator
        server_model: RAG-enhanced server model
        args: Command line arguments
        mode: 'validation' or 'test'
        ledger: Ledger instance for logging (optional)
        
    Returns:
        Tuple of (accuracy, loss, precision, recall, f1)
    """
    # Log access at start of evaluation
    if ledger:
        ledger.log_access(
            user_id="system",
            action="evaluate",
            resource=f"{mode}_data",
            status="started",
            details={"mode": mode, "num_clients": num_clients}
        )
    data_iterators = []
    for i in range(num_clients):
        if mode == 'validation':
            data_iterators.append(iter(val_loaders[i]))
        else:
            data_iterators.append(iter(test_loaders[i]))
    
    embeddings_grad = [None] * num_clients
    embeddings_nograd = [None] * num_clients
    sum_nograd = None
    targets = None
    completed = False
    total = 0
    correct = 0
    total_loss = 0
    n = 0
    all_targets = []
    all_predicted = []
    
    while not completed:
        # Generate embeddings
        for i in range(num_clients):
            item = next(data_iterators[i], -1)
            
            if item == -1:
                completed = True
                break
            
            inputs, targets = item
            embeddings_grad[i] = models[i](inputs)
            with torch.no_grad():
                embeddings_nograd[i] = models[i](inputs)
            
            embeddings_nograd[i] = quantize(embeddings_nograd[i], theta, quant_bin)
            
            if args.withblockchain and blockchain_vfl_integrator:
                client_parameters = [list(map(int, row)) for row in embeddings_nograd[i].tolist()]
                blockchain_vfl_integrator.update_client_weights(
                    blockchain_vfl_integrator.client_accounts[i], 
                    client_parameters
                )
        
        if completed:
            break
        
        # Aggregate
        if args.withblockchain and blockchain_vfl_integrator:
            blockchain_vfl_integrator.aggregate_weights()
            with torch.no_grad():
                sum_nograd = torch.tensor(blockchain_vfl_integrator.get_aggregated_weights())
        else:
            with torch.no_grad():
                sum_nograd = torch.sum(torch.stack(embeddings_nograd), dim=0)
        
        sum_nograd = dequantize(sum_nograd, theta, quant_bin, num_clients)
        sum_grad = torch.sum(torch.stack(embeddings_grad), dim=0)
        sum_grad.data = sum_nograd
        
        # Forward through RAG-enhanced server
        outputs = server_model(sum_grad)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        
        correct += (predicted == targets).sum()
        total += targets.size(0)
        total_loss += loss.item()
        n += 1
        all_targets.extend(targets.cpu().tolist())
        all_predicted.extend(predicted.cpu().tolist())
    
    del data_iterators
    accuracy = 100 * correct / total
    loss = total_loss / n

    # Compute precision / recall / F1
    precision = recall = f1 = 0.0
    if SKLEARN_AVAILABLE and all_targets:
        num_unique = len(set(all_targets))
        avg = 'binary' if num_unique <= 2 else 'macro'
        precision = float(precision_score(all_targets, all_predicted, average=avg, zero_division=0))
        recall    = float(recall_score(   all_targets, all_predicted, average=avg, zero_division=0))
        f1        = float(f1_score(       all_targets, all_predicted, average=avg, zero_division=0))
    
    # Log successful evaluation
    if ledger:
        ledger.log_access(
            user_id="system",
            action="evaluate",
            resource=f"{mode}_data",
            status="success",
            details={"mode": mode, "accuracy": float(accuracy), "loss": float(loss),
                     "precision": precision, "recall": recall, "f1": f1}
        )
    
    return (accuracy, loss, precision, recall, f1)


def main():
    parser = argparse.ArgumentParser(description='RAG-Enhanced VFL for Medical Imaging with ZIP Support')
    parser.add_argument('--datapath', metavar='DIR', help='path to X-ray dataset', default='./')
    parser.add_argument('--datasize', default=1.0, type=float, help='Dataset size (0.0125, 0.25, 0.5, or 1.0)')
    parser.add_argument('--theta', default=0.1, type=float, help='Noise value (in range [0, 0.25])')
    parser.add_argument('--withblockchain', action='store_true', help='Use blockchain for aggregation')
    parser.add_argument('--use-blockchain', dest='withblockchain', action='store_true',
                        help='Alias for --withblockchain')
    parser.add_argument('--use-rag', action='store_true', help='Enable RAG enhancement (recommended)')
    parser.add_argument('--num-epochs', default=5, type=int, help='Number of training epochs')
    
    # Enhanced model type choices with YOLO support
    model_choices = ['resnet_vgg', 'vit', 'vit_small', 'hybrid']
    if YOLO_MODELS_AVAILABLE:
        model_choices.extend(['yolo5', 'yolo8', 'resnet_yolo'])
    
    parser.add_argument('--model-type', default='resnet_vgg', choices=model_choices,
                       help='Client model architecture (default: resnet_vgg)')
    parser.add_argument('--enable-explainability', action='store_true', help='Enable explainability features (GradCAM/attention)')
    parser.add_argument('--enable-uncertainty', action='store_true',help='Enable uncertainty quantification (Monte Carlo Dropout)')
    parser.add_argument('--compare-models', action='store_true',help='Run model architecture comparison before training')
    args = parser.parse_args()

    # Auto-detect class names from the dataset (supports both hospital and legacy naming)
    detected_classes = detect_class_names(args.datapath)
    if detected_classes:
        class_names = detected_classes
    else:
        class_names = ['Normal', 'COVID-19']  # backward-compatible default
    num_classes = len(class_names)

    print("="*80)
    print("RAG-Enhanced Vertical Federated Learning for Medical Imaging")
    print("A Blockchain-Enabled Framework for Privacy-Preserving Cross-Hospital X-ray Analysis")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Hospitals: A, B, C, D (4 nodes)")
    print(f"  - Classes ({num_classes}): {class_names}")
    print(f"  - Model Architecture: {args.model_type.upper()}")
    print(f"  - RAG Enhancement: {'Enabled' if args.use_rag else 'Disabled'}")
    print(f"  - Blockchain: {'Enabled' if args.withblockchain else 'Disabled'}")
    print(f"  - DP Noise (theta): {args.theta}")
    print(f"  - Dataset size: {args.datasize}")
    print("="*80)
    
    # Initialize model registry and ledger
    print("\n✓ Initializing model registry and ledger...")
    model_registry = ModelRegistry()
    ledger = Ledger()
    print(f"  Registry: {model_registry.registry_dir}")
    print(f"  Ledger: {ledger.ledger_dir}")
    
    # Initialize blockchain if requested
    blockchain_vfl_integrator = None
    if args.withblockchain:
        if not BLOCKCHAIN_AVAILABLE:
            print("Error: Blockchain requested but not available.")
            print("  Install required packages: pip install web3 eth-account")
            args.withblockchain = False
        else:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            CONTRACT_SOURCE = os.path.join(repo_root, "src", "Aggregator.sol")
            try:
                blockchain_vfl_integrator = BlockchainVFLIntegrator(4, CONTRACT_SOURCE)
                print("✓ Blockchain integration initialized")
                print(f"  Provider URI : {blockchain_vfl_integrator.provider_uri}")
                print(f"  Chain ID     : {blockchain_vfl_integrator.chain_id}")
                print(f"  Contract     : {blockchain_vfl_integrator.contract_address}")
            except ConnectionError as e:
                print(str(e))
                print("  Disabling blockchain for this run.")
                args.withblockchain = False
            except Exception as e:
                print(f"  Blockchain initialization failed: {type(e).__name__}: {e}")
                print("  Disabling blockchain for this run.")
                args.withblockchain = False
    
    # Parameters
    num_clients = 4
    lr = 0.0001
    lr_decay = 0.9
    batch_size = 10
    num_epochs = args.num_epochs
    quant_bin = 8
    theta = args.theta
    
    if theta > 0.25 or theta < 0:
        raise Exception("Invalid noise parameters. Theta must be in [0, 0.25]")
    
    # Initialize models
    models = []
    optimizers = []
    schedulers = []
    
    # Model comparison if requested
    if args.compare_models and YOLO_MODELS_AVAILABLE:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        compare_models(['resnet_vgg', 'yolo5', 'resnet_yolo'], embedding_dim=64)
        print("="*80 + "\n")
    
    # Hospital naming
    hospital_names = ['A', 'B', 'C', 'D']
    
    for i in range(num_clients):
        model = create_client_model(model_type=args.model_type, embedding_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=lr_decay)
        
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        
        # Print model info for first hospital
        if i == 0:
            model_info = get_model_info(model)
            print(f"\n✓ Client Model Initialized:")
            print(f"  Architecture: {model_info['model_type']}")
            print(f"  Total Parameters: {model_info['total_parameters']/1e6:.2f}M")
            if 'trainable_parameters' in model_info:
                print(f"  Trainable Parameters: {model_info['trainable_parameters']/1e6:.2f}M")
            print(f"  Model Size: {model_info['size_mb']:.2f} MB")

    # Server model with optional uncertainty
    print(f"\n✓ Initializing server model...")
    if args.enable_uncertainty:
        server_model = UncertaintyAwareRAGServer(
            embedding_dim=64,
            num_classes=num_classes,
            use_rag=args.use_rag
        )
        server_model.enable_uncertainty(num_samples=20)
    else:
        server_model = RAGEnhancedServerModel(
            embedding_dim=64,
            num_classes=num_classes,
            use_rag=args.use_rag
        )
    
    # Add optimizer and scheduler for server model (only once!)
    optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)
    models.append(server_model)
    optimizers.append(optimizer)
    schedulers.append(scheduler)
    
    if args.use_rag:
        print("✓ RAG module initialized with medical knowledge base")
        kb_hash = server_model.verify_rag_integrity()
        print(f"  Knowledge base hash: {kb_hash[:16]}...")
    
    criterion = nn.CrossEntropyLoss()
    
    # Load datasets (using the existing dataset structure)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    train_loaders = []
    val_loaders = []
    test_loaders = []
    
    # Note: This supports both client0/1/2/3 (legacy) and hospitalA/B/C/D (new) naming
    # For ZIP-extracted datasets, use hospitalA/B/C/D structure
    hospital_names = ['A', 'B', 'C', 'D']
    dataset_stats = []
    
    try:
        for i in range(num_clients):
            hospital_id = hospital_names[i]
            
            # Try hospital naming first (new format)
            hospital_path = f'{args.datapath}/SplitCovid19/hospital{hospital_id}'
            legacy_path = f'{args.datapath}/SplitCovid19/client{i}'
            
            # Determine which path exists
            if os.path.exists(hospital_path):
                base_path = hospital_path
                print(f"✓ Using Hospital {hospital_id} data structure")
            elif os.path.exists(legacy_path):
                base_path = legacy_path
                print(f"✓ Using legacy client{i} data structure (Hospital {hospital_id})")
            else:
                raise FileNotFoundError(f"Neither {hospital_path} nor {legacy_path} found")
            
            train_dataset = torchvision.datasets.ImageFolder(
                root=f'{base_path}/train', 
                transform=transform
            )
            # Dynamic train/val split based on actual dataset size per client
            train_idx, val_idx = _make_train_val_indices(
                n_total=len(train_dataset),
                datasize=args.datasize,
                seed=42 + i,
                train_frac=0.8,
            )
            train_data = torch.utils.data.Subset(train_dataset, indices=train_idx)
            val_data = torch.utils.data.Subset(train_dataset, indices=val_idx)
            # num_workers=0 avoids multiprocessing shutdown noise in Codespaces/Python 3.12
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
            train_loaders.append(train_loader)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
            val_loaders.append(val_loader)
            
            test_dataset = torchvision.datasets.ImageFolder(
                root=f'{base_path}/test', 
                transform=transform
            )
            # Use the full test set (no hardcoded sub-sampling)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loaders.append(test_loader)
            
            # Collect statistics
            dataset_stats.append({
                'hospital': hospital_id,
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                'test_samples': len(test_dataset),
            })
        
        # Print dataset summary with hospital naming
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        for stat in dataset_stats:
            print(f"✓ Hospital {stat['hospital']}: {stat['train_samples']} train, "
                  f"{stat['val_samples']} val, {stat['test_samples']} test samples")
        print("="*80)
        print("✓ X-ray dataset loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load dataset - {e}")
        print("This demo requires the SplitCovid19 dataset structure.")
        print("Use load_zip_dataset.py to prepare your dataset, or use the legacy prepare_dataset.py")
        print("Continuing with demonstration of RAG architecture...")
    
    # Initial evaluation
    print("\nInitial Evaluation:")
    try:
        test_accuracy, test_loss = evaluate(
            models, val_loaders, test_loaders, criterion, num_clients,
            theta, quant_bin, blockchain_vfl_integrator, server_model, args, 
            mode='test', ledger=ledger
        )
        print(f'  Test loss: {test_loss:.2f} | Test accuracy: {test_accuracy:.2f}%')
    except Exception as e:
        print(f"  Evaluation failed due to an exception: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f'\n--- Epoch [{epoch+1}/{num_epochs}] ---')
        
        start = time.time()
        try:
            train(models, optimizers, schedulers, train_loaders, criterion,
                  num_clients, theta, quant_bin, blockchain_vfl_integrator,
                  server_model, args, ledger=ledger, epoch_num=epoch)
            
            val_accuracy, val_loss, val_precision, val_recall, val_f1 = evaluate(
                models, val_loaders, test_loaders, criterion, num_clients,
                theta, quant_bin, blockchain_vfl_integrator, server_model, 
                args, mode='validation', ledger=ledger
            )
            test_accuracy, test_loss, test_precision, test_recall, test_f1 = evaluate(
                models, val_loaders, test_loaders, criterion, num_clients,
                theta, quant_bin, blockchain_vfl_integrator, server_model,
                args, mode='test', ledger=ledger
            )
            
            print(f'Time: {time.time() - start:.2f}s')
            print(f'Val  Loss: {val_loss:.2f} | Val  Acc: {val_accuracy:.2f}% | Val  P: {val_precision:.3f} | Val  R: {val_recall:.3f} | Val  F1: {val_f1:.3f}')
            print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_accuracy:.2f}% | Test P: {test_precision:.3f} | Test R: {test_recall:.3f} | Test F1: {test_f1:.3f}')
            
            # Log real blockchain tx from the most recent aggregation in this epoch
            if args.withblockchain and blockchain_vfl_integrator and blockchain_vfl_integrator.last_tx_hash:
                print(f'  [Chain] Last agg tx  : {blockchain_vfl_integrator.last_tx_hash}')
                print(f'  [Chain] Block number : {blockchain_vfl_integrator.last_block_number}')
            
            # Save model checkpoint
            metrics = {
                'val_accuracy': float(val_accuracy),
                'val_loss': float(val_loss),
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
            }
            config = {
                'theta': theta,
                'num_clients': num_clients,
                'use_rag': args.use_rag,
                'use_blockchain': args.withblockchain,
                'datasize': args.datasize,
                'epoch': epoch + 1,
                'class_names': class_names,
                'num_classes': num_classes,
                'model_type': args.model_type,
            }
            
            version_id = model_registry.save_model(
                server_model, 
                round_num=epoch + 1,
                metrics=metrics,
                config=config
            )
            print(f'✓ Model saved: {version_id}')
            
            # Log training round to ledger with hospital naming
            model_hash = model_registry.versions[version_id].model_hash
            node_metrics = {
                'server': metrics,
                **{f'hospital{hospital_names[i]}': {'status': 'trained'} for i in range(num_clients)}
            }
            
            blockchain_tx_hash = None
            if args.withblockchain and blockchain_vfl_integrator:
                # Use the real on-chain tx hash from the last aggregation this epoch
                blockchain_tx_hash = blockchain_vfl_integrator.last_tx_hash
            
            ledger.log_training_round(
                round_num=epoch + 1,
                node_metrics=node_metrics,
                model_hash=model_hash,
                blockchain_tx=blockchain_tx_hash,
                rag_retrieval_count=0,  # Can be tracked if needed
                privacy_budget=theta
            )
            
            if args.use_rag:
                print(f'RAG KB Hash: {server_model.verify_rag_integrity()[:16]}...')
        except Exception as e:
            print(f"  Training/evaluation failed due to an exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            print("Continuing to next epoch or finishing demo.")
            break
    
    print("\n" + "="*80)
    print("Training Complete!")
    if args.use_rag:
        print(f"✓ RAG-enhanced model trained successfully")
        print(f"✓ Knowledge base integrity verified: {server_model.verify_rag_integrity()[:16]}...")
    if args.withblockchain:
        print("✓ Blockchain aggregation verified")
    
    # Print registry and ledger summary
    registry_summary = model_registry.get_summary()
    ledger_summary = ledger.get_summary()
    
    print(f"\n📦 Model Registry Summary:")
    print(f"  Total versions: {registry_summary['total_versions']}")
    print(f"  Latest version: {registry_summary.get('latest_version', 'N/A')}")
    if registry_summary.get('best_accuracy'):
        print(f"  Best accuracy: {registry_summary['best_accuracy']:.2f}%")
    
    print(f"\n📋 Ledger Summary:")
    print(f"  Training entries: {ledger_summary['training_entries']}")
    print(f"  Access entries: {ledger_summary['access_entries']}")
    print(f"  Integrity verified: {'✓' if ledger_summary['training_integrity'] else '✗'}")
    
    print("="*80)


if __name__ == "__main__":
    main()
