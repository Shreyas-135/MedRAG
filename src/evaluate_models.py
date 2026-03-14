"""
Model Comparison and Evaluation Script
Compares ResNet, VGG, ViT, and Hybrid models for federated learning
"""

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
