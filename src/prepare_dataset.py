#!/usr/bin/env python3
"""
Dataset Preparation Script for MedRAG Federated Learning

This script downloads and prepares X-ray datasets for the federated learning
component of MedRAG. It creates a directory structure compatible with the
repository's requirements.

Directory Structure Created:
    your_dataset_path/
    ├── SplitCovid19/
    │   ├── client0/
    │   │   ├── train/
    │   │   │   ├── covid/
    │   │   │   └── normal/
    │   │   └── test/
    │   │       ├── covid/
    │   │       └── normal/
    │   ├── client1/
    │   ├── client2/
    │   └── client3/

Usage:
    # Basic usage - downloads datasets from Kaggle
    python prepare_dataset.py --output-dir ./data --kaggle-username YOUR_USERNAME
    
    # Use existing local datasets
    python prepare_dataset.py --output-dir ./data --use-local-data --local-data-dir ./raw_datasets
    
    # Custom distribution
    python prepare_dataset.py --output-dir ./data --num-clients 4 --train-split 0.8

Requirements:
    - kaggle API credentials (kaggle.json) in ~/.kaggle/ for downloading
    - Or pre-downloaded datasets in the local directory
    - PIL/Pillow for image processing
    - numpy for data manipulation
"""

import os
import sys
import argparse
import shutil
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
import zipfile
import tarfile

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


class DatasetPreparer:
    """Handles downloading and preparing datasets for federated learning."""
    
    # Default image dimensions for synthetic sample images
    DEFAULT_IMAGE_WIDTH = 224
    DEFAULT_IMAGE_HEIGHT = 224
    DEFAULT_IMAGE_CHANNELS = 3
    
    def __init__(self, output_dir: str, num_clients: int = 4, train_split: float = 0.8):
        """
        Initialize the dataset preparer.
        
        Args:
            output_dir: Output directory for prepared dataset
            num_clients: Number of federated learning clients (default: 4)
            train_split: Ratio of training data (default: 0.8)
        """
        self.output_dir = Path(output_dir)
        self.num_clients = num_clients
        self.train_split = train_split
        self.base_dir = self.output_dir / "SplitCovid19"
        
        # Dataset URLs and information
        self.datasets = {
            'covid19': {
                'kaggle_dataset': 'tawsifurrahman/covid19-radiography-database',
                'description': 'COVID-19 Radiography Database'
            },
            'pneumonia': {
                'kaggle_dataset': 'paultimothymooney/chest-xray-pneumonia',
                'description': 'Chest X-Ray Pneumonia Dataset'
            },
            'tuberculosis': {
                'kaggle_dataset': 'tawsifurrahman/tuberculosis-tb-chest-xray-dataset',
                'description': 'Tuberculosis Chest X-Ray Dataset'
            }
        }
        
    def setup_directories(self):
        """Create the required directory structure for all clients."""
        print("\n" + "="*70)
        print("Setting up directory structure...")
        print("="*70)
        
        for client_id in range(self.num_clients):
            client_dir = self.base_dir / f"client{client_id}"
            for split in ['train', 'test']:
                for label in ['covid', 'normal']:
                    dir_path = client_dir / split / label
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"✓ Created: {dir_path}")
        
        print(f"\n✓ Directory structure created successfully at: {self.base_dir}")
        
    def check_kaggle_setup(self) -> bool:
        """Check if Kaggle API is properly configured."""
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            return False
        
        try:
            import kaggle
            return True
        except ImportError:
            return False
            
    def download_from_kaggle(self, dataset_name: str, download_dir: Path) -> bool:
        """
        Download dataset from Kaggle.
        
        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'username/dataset-name')
            download_dir: Directory to download the dataset to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import kaggle
            
            print(f"\nDownloading {dataset_name}...")
            download_dir.mkdir(parents=True, exist_ok=True)
            
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(download_dir),
                unzip=True,
                quiet=False
            )
            
            print(f"✓ Downloaded: {dataset_name}")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading {dataset_name}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path):
        """Extract zip or tar archives."""
        print(f"Extracting {archive_path.name}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        
        print(f"✓ Extracted to: {extract_to}")
    
    def find_images_in_directory(self, directory: Path, extensions: List[str] = None) -> List[Path]:
        """
        Recursively find all image files in a directory.
        
        Args:
            directory: Root directory to search
            extensions: List of valid image extensions (default: common image formats)
            
        Returns:
            List of image file paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        images = []
        for ext in extensions:
            images.extend(directory.rglob(f"*{ext}"))
            images.extend(directory.rglob(f"*{ext.upper()}"))
        
        return images
    
    def validate_and_load_image(self, image_path: Path) -> bool:
        """
        Validate that an image can be loaded and is valid.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            # Open and load the image to verify it's valid
            # Note: we don't use verify() as it renders the image unusable
            with Image.open(image_path) as img:
                img.load()  # Force loading to detect truncated/corrupt images
            return True
        except Exception:
            return False
    
    def categorize_images_from_local(self, local_data_dir: Path) -> Dict[str, List[Path]]:
        """
        Categorize images from local directory into covid and normal classes.
        
        Args:
            local_data_dir: Directory containing raw datasets
            
        Returns:
            Dictionary with 'covid' and 'normal' keys containing image paths
        """
        print("\n" + "="*70)
        print("Categorizing images from local directory...")
        print("="*70)
        
        categorized = {'covid': [], 'normal': []}
        
        # Search for images in the local directory
        all_images = self.find_images_in_directory(local_data_dir)
        
        print(f"Found {len(all_images)} images in {local_data_dir}")
        
        # Categorize based on directory names and file names
        for img_path in all_images:
            path_lower = str(img_path).lower()
            
            # Check if image is valid
            if not self.validate_and_load_image(img_path):
                print(f"⚠ Skipping invalid image: {img_path.name}")
                continue
            
            # Categorize based on path/filename
            if any(keyword in path_lower for keyword in ['covid', 'covid-19', 'covid19', 'sars-cov-2']):
                categorized['covid'].append(img_path)
            elif any(keyword in path_lower for keyword in ['normal', 'healthy', 'negative']):
                categorized['normal'].append(img_path)
            elif any(keyword in path_lower for keyword in ['pneumonia', 'tuberculosis', 'tb']):
                # For pneumonia and TB X-rays, categorize them as "normal" (non-COVID)
                # since they don't have COVID-19, even though they have other pathologies.
                # This is appropriate for binary COVID vs. non-COVID classification.
                categorized['normal'].append(img_path)
        
        print(f"\n✓ Categorized {len(categorized['covid'])} COVID images")
        print(f"✓ Categorized {len(categorized['normal'])} normal images")
        
        return categorized
    
    def download_and_organize_datasets(self, use_local: bool = False, local_dir: Path = None) -> Dict[str, List[Path]]:
        """
        Download datasets from Kaggle or use local datasets, and organize them.
        
        Args:
            use_local: If True, use local datasets instead of downloading
            local_dir: Directory containing local datasets
            
        Returns:
            Dictionary with categorized image paths
        """
        if use_local and local_dir:
            return self.categorize_images_from_local(local_dir)
        
        # Check Kaggle setup
        if not self.check_kaggle_setup():
            print("\n" + "="*70)
            print("ERROR: Kaggle API not configured")
            print("="*70)
            print("\nTo download datasets from Kaggle, you need to:")
            print("1. Install kaggle package: pip install kaggle")
            print("2. Get your API credentials from https://www.kaggle.com/account")
            print("3. Place kaggle.json in ~/.kaggle/")
            print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
            print("\nAlternatively, use --use-local-data with pre-downloaded datasets")
            sys.exit(1)
        
        # Create temporary download directory
        download_dir = self.output_dir / "downloads"
        download_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("Downloading datasets from Kaggle...")
        print("="*70)
        
        # Download datasets
        for dataset_key, dataset_info in self.datasets.items():
            dataset_download_dir = download_dir / dataset_key
            success = self.download_from_kaggle(
                dataset_info['kaggle_dataset'],
                dataset_download_dir
            )
            if not success:
                print(f"⚠ Warning: Failed to download {dataset_key}")
        
        # Now categorize the downloaded images
        return self.categorize_images_from_local(download_dir)
    
    def distribute_images_to_clients(self, categorized_images: Dict[str, List[Path]]):
        """
        Distribute images across clients with train/test split.
        
        Args:
            categorized_images: Dictionary with 'covid' and 'normal' image paths
        """
        print("\n" + "="*70)
        print("Distributing images to clients...")
        print("="*70)
        
        for label, images in categorized_images.items():
            if not images:
                print(f"⚠ Warning: No images found for label '{label}'")
                continue
            
            # Shuffle images
            random.shuffle(images)
            
            # Split into train and test
            split_idx = int(len(images) * self.train_split)
            train_images = images[:split_idx]
            test_images = images[split_idx:]
            
            print(f"\n{label.upper()} images:")
            print(f"  Total: {len(images)}")
            print(f"  Train: {len(train_images)}")
            print(f"  Test:  {len(test_images)}")
            
            # Distribute train images across clients
            self._distribute_images_to_clients_helper(train_images, label, 'train')
            
            # Distribute test images across clients
            self._distribute_images_to_clients_helper(test_images, label, 'test')
        
        print("\n✓ Image distribution complete!")
    
    def _distribute_images_to_clients_helper(self, images: List[Path], label: str, split: str):
        """
        Helper method to distribute images to clients for a specific split.
        
        Args:
            images: List of image paths to distribute
            label: Image label ('covid' or 'normal')
            split: 'train' or 'test'
        """
        images_per_client = len(images) // self.num_clients
        for client_id in range(self.num_clients):
            start_idx = client_id * images_per_client
            # Last client gets any remaining images
            end_idx = start_idx + images_per_client if client_id < self.num_clients - 1 else len(images)
            client_images = images[start_idx:end_idx]
            
            dest_dir = self.base_dir / f"client{client_id}" / split / label
            self.copy_images_to_directory(client_images, dest_dir, client_id, split, label)
    
    def copy_images_to_directory(self, images: List[Path], dest_dir: Path, 
                                  client_id: int, split: str, label: str):
        """
        Copy images to destination directory with progress tracking.
        
        Args:
            images: List of image paths to copy
            dest_dir: Destination directory
            client_id: Client ID
            split: 'train' or 'test'
            label: Image label ('covid' or 'normal')
        """
        for i, img_path in enumerate(images):
            # Create unique filename
            dest_path = dest_dir / f"{label}_{client_id}_{split}_{i}{img_path.suffix}"
            
            try:
                shutil.copy2(img_path, dest_path)
            except Exception as e:
                print(f"⚠ Error copying {img_path.name}: {e}")
        
        print(f"  ✓ Client {client_id} {split}/{label}: {len(images)} images")
    
    def generate_dataset_info(self):
        """Generate a JSON file with dataset information and statistics."""
        info = {
            'num_clients': self.num_clients,
            'train_split': self.train_split,
            'structure': 'SplitCovid19/client{0-' + str(self.num_clients-1) + '}/{train,test}/{covid,normal}/',
            'clients': {}
        }
        
        # Collect statistics for each client
        for client_id in range(self.num_clients):
            client_info = {'train': {}, 'test': {}}
            
            for split in ['train', 'test']:
                for label in ['covid', 'normal']:
                    dir_path = self.base_dir / f"client{client_id}" / split / label
                    image_count = len(list(dir_path.glob('*.*')))
                    client_info[split][label] = image_count
            
            info['clients'][f'client{client_id}'] = client_info
        
        # Save to JSON
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✓ Dataset info saved to: {info_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        for client_id in range(self.num_clients):
            client_info = info['clients'][f'client{client_id}']
            print(f"\nClient {client_id}:")
            print(f"  Train: {client_info['train']['covid']} COVID, {client_info['train']['normal']} normal")
            print(f"  Test:  {client_info['test']['covid']} COVID, {client_info['test']['normal']} normal")
    
    def create_sample_dataset(self, num_samples_per_class: int = 50):
        """
        Create a small sample dataset for testing purposes.
        
        Args:
            num_samples_per_class: Number of sample images per class
        """
        print("\n" + "="*70)
        print("Creating sample dataset for testing...")
        print("="*70)
        
        # Create simple synthetic images
        for client_id in range(self.num_clients):
            for split in ['train', 'test']:
                samples = num_samples_per_class if split == 'train' else num_samples_per_class // 2
                
                for label in ['covid', 'normal']:
                    dest_dir = self.base_dir / f"client{client_id}" / split / label
                    
                    for i in range(samples):
                        # Create a simple synthetic image
                        # COVID images: darker with random patterns
                        # Normal images: lighter with random patterns
                        img_shape = (self.DEFAULT_IMAGE_HEIGHT, self.DEFAULT_IMAGE_WIDTH, 
                                   self.DEFAULT_IMAGE_CHANNELS)
                        if label == 'covid':
                            img_array = np.random.randint(50, 150, img_shape, dtype=np.uint8)
                        else:
                            img_array = np.random.randint(150, 250, img_shape, dtype=np.uint8)
                        
                        img = Image.fromarray(img_array)
                        img_path = dest_dir / f"{label}_{client_id}_{split}_{i}.png"
                        img.save(img_path)
                    
                    print(f"  ✓ Client {client_id} {split}/{label}: {samples} images")
        
        print("\n✓ Sample dataset created successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare X-ray datasets for MedRAG federated learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download from Kaggle and prepare dataset
    python prepare_dataset.py --output-dir ./data
    
    # Use local datasets
    python prepare_dataset.py --output-dir ./data --use-local-data --local-data-dir ./raw_datasets
    
    # Create a small sample dataset for testing
    python prepare_dataset.py --output-dir ./data --create-sample --sample-size 50
    
    # Custom configuration
    python prepare_dataset.py --output-dir ./data --num-clients 4 --train-split 0.8
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./dataset',
        help='Output directory for prepared dataset (default: ./dataset)'
    )
    
    parser.add_argument(
        '--num-clients',
        type=int,
        default=4,
        help='Number of federated learning clients (default: 4)'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Ratio of training data (default: 0.8)'
    )
    
    parser.add_argument(
        '--use-local-data',
        action='store_true',
        help='Use local datasets instead of downloading from Kaggle'
    )
    
    parser.add_argument(
        '--local-data-dir',
        type=str,
        help='Directory containing local datasets (required if --use-local-data is set)'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create a small sample dataset for testing (no download needed)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50,
        help='Number of sample images per class when creating sample dataset (default: 50)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_local_data and not args.local_data_dir:
        parser.error("--local-data-dir is required when --use-local-data is set")
    
    if args.use_local_data and args.create_sample:
        parser.error("Cannot use both --use-local-data and --create-sample")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Print header
    print("\n" + "="*70)
    print("MedRAG Dataset Preparation Tool")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Train/test split: {args.train_split:.1%} / {1-args.train_split:.1%}")
    
    if args.create_sample:
        print(f"Mode: Creating sample dataset ({args.sample_size} images per class)")
    elif args.use_local_data:
        print(f"Mode: Using local data from {args.local_data_dir}")
    else:
        print("Mode: Downloading from Kaggle")
    
    # Initialize preparer
    preparer = DatasetPreparer(
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        train_split=args.train_split
    )
    
    # Setup directory structure
    preparer.setup_directories()
    
    # Process datasets
    if args.create_sample:
        # Create sample dataset
        preparer.create_sample_dataset(num_samples_per_class=args.sample_size)
    else:
        # Download or load local datasets
        local_dir = Path(args.local_data_dir) if args.local_data_dir else None
        categorized_images = preparer.download_and_organize_datasets(
            use_local=args.use_local_data,
            local_dir=local_dir
        )
        
        # Check if we have enough images
        if not categorized_images['covid'] and not categorized_images['normal']:
            print("\n" + "="*70)
            print("ERROR: No images found!")
            print("="*70)
            print("\nPlease ensure:")
            print("1. Your local directory contains X-ray images, OR")
            print("2. Kaggle API is properly configured for downloading")
            print("\nAlternatively, create a sample dataset with --create-sample")
            sys.exit(1)
        
        # Distribute images to clients
        preparer.distribute_images_to_clients(categorized_images)
    
    # Generate dataset information
    preparer.generate_dataset_info()
    
    # Print completion message
    print("\n" + "="*70)
    print("✓ DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nYour dataset is ready at: {preparer.base_dir}")
    print("\nTo use this dataset with MedRAG:")
    print(f"  cd src")
    print(f"  python demo_rag_vfl.py --datapath {args.output_dir} --use-rag")
    print(f"  python demo_rag_vfl.py --datapath {args.output_dir} --use-rag --withblockchain")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
