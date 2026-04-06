"""
Model Comparison and Evaluation Script
Compares ResNet, VGG, ViT, and Hybrid models for federated learning
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from models import create_client_model, get_model_info


def benchmark_model(model, dataloader, device='cpu', num_batches=None):
    """
    Benchmark model performance
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break
                
            images, labels = images.to(device), labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Calculate accuracy (assuming outputs are embeddings, not logits)
            # For proper evaluation, you'd need the full pipeline with server model
            total += labels.size(0)
    
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
    return {
        'avg_inference_time': avg_inference_time,
        'total_samples': total,
        'throughput': total / sum(inference_times) if inference_times else 0
    }


def compare_architectures(device='cpu', batch_size=16):
    """
    Compare different model architectures
    
    Returns:
        DataFrame with comparison results
    """
    models_to_test = {
        'ResNet50+VGG19': 'resnet_vgg',
        'ViT-Base':  'vit',
        'ViT-Small': 'vit_small',
        'Hybrid (ResNet+ViT)': 'hybrid',
    }
    
    results = []
    
    print("="*80)
    print("Model Architecture Comparison")
    print("="*80)
    
    for name, model_type in models_to_test.items():
        print(f"\n📊 Evaluating {name}...")
        
        try:
            model = create_client_model(model_type=model_type, embedding_dim=64)
            model_info = get_model_info(model)
            
            # Create dummy data for benchmarking
            dummy_data = torch.randn(batch_size, 3, 224, 224)
            dummy_labels = torch. randint(0, 2, (batch_size,))
            dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
            dummy_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False)
            
            # Benchmark
            metrics = benchmark_model(model, dummy_loader, device, num_batches=10)
            
            results.append({
                'Model': name,
                'Parameters (M)': model_info['total_parameters'] / 1e6,
                'Model Size (MB)': model_info['size_mb'],
                'Inference Time (ms)': metrics['avg_inference_time'] * 1000,
                'Throughput (samples/s)': metrics['throughput'],
                'Type': model_info['model_type']
            })
            
            print(f"  ✓ Parameters: {model_info['total_parameters']/1e6:.2f}M")
            print(f"  ✓ Size: {model_info['size_mb']:.2f} MB")
            print(f"  ✓ Inference:  {metrics['avg_inference_time']*1000:.2f} ms")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    return pd.DataFrame(results)


def plot_comparison(df, output_path='model_comparison.png'):
    """Create comparison plots"""
    if df.empty:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Architecture Comparison', fontsize=16, fontweight='bold')
    
    # Parameters
    axes[0, 0].barh(df['Model'], df['Parameters (M)'], color='skyblue')
    axes[0, 0].set_xlabel('Parameters (Millions)')
    axes[0, 0].set_title('Model Parameters')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Inference Time
    axes[0, 1].barh(df['Model'], df['Inference Time (ms)'], color='lightcoral')
    axes[0, 1].set_xlabel('Inference Time (ms)')
    axes[0, 1].set_title('Inference Speed (Lower is Better)')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Model Size
    axes[1, 0].barh(df['Model'], df['Model Size (MB)'], color='lightgreen')
    axes[1, 0].set_xlabel('Model Size (MB)')
    axes[1, 0].set_title('Memory Footprint')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Throughput
    axes[1, 1].barh(df['Model'], df['Throughput (samples/s)'], color='gold')
    axes[1, 1].set_xlabel('Throughput (samples/second)')
    axes[1, 1].set_title('Processing Throughput (Higher is Better)')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {output_path}")


def calibration_benchmark(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    hospital_id: str = "unknown",
    device: str = "cpu",
    output_dir: str = "outputs",
) -> dict:
    """
    Calibrate *model* on *val_loader*, evaluate calibration on *test_loader*,
    and produce a reliability diagram.

    This is a convenience wrapper around ``TemperatureScaler``,
    ``compute_calibration_metrics``, and ``plot_reliability_diagram`` from
    ``src/calibration.py``.

    Args:
        model: PyTorch model returning raw logits.
        val_loader: Held-out calibration (validation) DataLoader.
        test_loader: Test DataLoader for reporting final metrics.
        hospital_id: Hospital / client identifier for labelling outputs.
        device: 'cpu' or 'cuda'.
        output_dir: Directory for the reliability diagram PNG.

    Returns:
        Dict with keys: ``temperature``, ``pre_calibration``,
        ``post_calibration``, ``hospital_id``.
        Returns an empty dict if the calibration module is unavailable.
    """
    try:
        from calibration import (
            TemperatureScaler,
            compute_calibration_metrics,
            plot_reliability_diagram,
        )
    except ImportError:
        print("Warning: calibration module not available — skipping calibration benchmark.")
        return {}

    import numpy as np

    print(f"\n{'='*60}")
    print(f"Calibration Benchmark — {hospital_id}")
    print(f"{'='*60}")

    # ---- Fit temperature on validation set ---------------------------------
    scaler = TemperatureScaler(model)
    optimal_T = scaler.fit(val_loader, device=device)
    print(f"  Optimal temperature T = {optimal_T:.4f}")

    # ---- Collect test logits / labels for metric evaluation ----------------
    model.eval()
    model.to(device)
    all_logits_pre, all_logits_post, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits_pre.append(logits.cpu().numpy())
            all_logits_post.append((logits / scaler.temperature).cpu().numpy())
            all_labels.append(labels.numpy())

    logits_pre = np.concatenate(all_logits_pre, axis=0)
    logits_post = np.concatenate(all_logits_post, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    pre = compute_calibration_metrics(logits_pre, labels_arr)
    post = compute_calibration_metrics(logits_post, labels_arr)

    print(f"  Pre-calibration  ECE = {pre['ece']:.4f}  MCE = {pre['mce']:.4f}")
    print(f"  Post-calibration ECE = {post['ece']:.4f}  MCE = {post['mce']:.4f}")

    # ---- Reliability diagram -----------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{hospital_id}_reliability.png")
    plot_reliability_diagram(
        logits_post,
        labels_arr,
        title=f"Reliability Diagram — {hospital_id} (T={optimal_T:.3f})",
        save_path=save_path,
    )

    return {
        "hospital_id": hospital_id,
        "temperature": optimal_T,
        "pre_calibration": pre,
        "post_calibration": post,
    }


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Compare model architectures')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for benchmarking')
    parser.add_argument('--output', type=str, default='model_comparison. png', help='Output plot path')
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    # Run comparison
    results_df = compare_architectures(device=device, batch_size=args.batch_size)
    
    # Display results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('model_comparison_results.csv', index=False)
    print(f"\n✓ Results saved to model_comparison_results.csv")
    
    # Plot comparison
    plot_comparison(results_df, output_path=args.output)
    
    print("\n✅ Evaluation complete!")
