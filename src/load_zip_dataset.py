#!/usr/bin/env python3
"""
ZIP Dataset Loader for MedRAG Federated Learning

This script extracts X-ray images from ZIP files and organizes them into
a federated learning structure with automatic distribution across hospitals.

Features:
- Accepts ZIP files with X-ray images (jpg, png, jpeg formats)
- Categorizes images by filename/folder (covid, normal, pneumonia, etc.)
- Automatically distributes images across 4 hospitals (A, B, C, D)
- Creates train/test splits (default 80/20)
- Supports binary and multi-class classification
- Provides summary statistics

Directory Structure Created:
    data/
    └── SplitCovid19/
        ├── hospitalA/
        │   ├── train/
        │   │   ├── covid/
        │   │   └── normal/
        │   └── test/
        │       ├── covid/
        │       └── normal/
        ├── hospitalB/
        ├── hospitalC/
        └── hospitalD/

Usage:
    python src/load_zip_dataset.py --zip-file /path/to/xray_dataset.zip --output-dir ./data --num-hospitals 4
"""

import os
import sys
import argparse
import zipfile
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not available. Install with: pip install tqdm")
    tqdm = lambda x, **kwargs: x

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


class ZipDatasetLoader:
    """Handles extraction and organization of X-ray datasets from ZIP files."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    # Common X-ray condition keywords for automatic classification
    CONDITION_KEYWORDS = {
        'covid': ['covid', 'covid19', 'covid-19', 'coronavirus'],
        'normal': ['normal', 'healthy'],
        'pneumonia': ['pneumonia', 'bacterial', 'viral'],
        'tuberculosis': ['tb', 'tuberculosis'],
        'lung_opacity': ['opacity', 'effusion'],
    }
    
    def __init__(self, zip_file: str, output_dir: str, num_hospitals: int = 4, 
                 train_split: float = 0.8, binary_classification: bool = False):
        """
        Initialize the ZIP dataset loader.
        
        Args:
            zip_file: Path to ZIP file containing X-ray images
            output_dir: Output directory for organized dataset
            num_hospitals: Number of hospitals to distribute data across (default: 4)
            train_split: Ratio of training data (default: 0.8)
            binary_classification: If True, only use 'covid' and 'normal' classes
        """
        self.zip_file = Path(zip_file)
        self.output_dir = Path(output_dir)
        self.num_hospitals = num_hospitals
        self.train_split = train_split
        self.binary_classification = binary_classification
        self.base_dir = self.output_dir / "SplitCovid19"
        
        # Hospital naming: A, B, C, D
        self.hospital_names = [chr(65 + i) for i in range(num_hospitals)]  # ['A', 'B', 'C', 'D']
        
        # Statistics
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def extract_zip(self, temp_dir: Path) -> Path:
        """
        Extract ZIP file to temporary directory.
        
        Args:
            temp_dir: Temporary directory for extraction
            
        Returns:
            Path to extracted contents
        """
        print(f"📦 Extracting ZIP file: {self.zip_file.name}")
        
        if not self.zip_file.exists():
            raise FileNotFoundError(f"ZIP file not found: {self.zip_file}")
        
        with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
            # Get total file count for progress bar
            total_files = len(zip_ref.namelist())
            
            # Extract with progress bar
            for member in tqdm(zip_ref.namelist(), desc="Extracting", total=total_files):
                try:
                    zip_ref.extract(member, temp_dir)
                except Exception as e:
                    print(f"Warning: Could not extract {member}: {e}")
        
        print(f"✓ Extraction complete")
        return temp_dir
    
    def classify_image(self, filename: str, parent_folder: str) -> str:
        """
        Classify image based on filename or parent folder.
        
        Args:
            filename: Image filename
            parent_folder: Parent folder name
            
        Returns:
            Condition label (covid, normal, etc.)
        """
        # Check filename and folder for keywords
        search_text = f"{filename.lower()} {parent_folder.lower()}"
        
        for condition, keywords in self.CONDITION_KEYWORDS.items():
            if any(keyword in search_text for keyword in keywords):
                return condition
        
        # Default to 'unknown' if no match
        return 'unknown'
    
    def collect_images(self, extracted_dir: Path) -> Dict[str, List[Path]]:
        """
        Collect and categorize all images from extracted directory.
        
        Args:
            extracted_dir: Path to extracted contents
            
        Returns:
            Dictionary mapping condition labels to image paths
        """
        print("\n🔍 Scanning for X-ray images...")
        
        images_by_class = defaultdict(list)
        
        # Walk through all files
        for root, dirs, files in os.walk(extracted_dir):
            parent_folder = Path(root).name
            
            for file in files:
                file_path = Path(root) / file
                
                # Check if it's a supported image format
                if file_path.suffix in self.SUPPORTED_FORMATS:
                    # Verify it's a valid image
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                        
                        # Classify the image
                        condition = self.classify_image(file, parent_folder)
                        images_by_class[condition].append(file_path)
                        
                    except Exception as e:
                        print(f"Warning: Invalid image {file_path}: {e}")
        
        # Print summary
        print(f"\n📊 Found images by class:")
        for condition, images in sorted(images_by_class.items()):
            print(f"   - {condition}: {len(images)} images")
        
        # Filter for binary classification if requested
        if self.binary_classification:
            filtered_images = {
                'covid': images_by_class['covid'],
                'normal': images_by_class['normal']
            }
            print(f"\n⚠️  Binary classification mode: Using only 'covid' and 'normal' classes")
            return filtered_images
        
        return dict(images_by_class)
    
    def distribute_images(self, images_by_class: Dict[str, List[Path]]):
        """
        Distribute images across hospitals with train/test splits.
        
        Args:
            images_by_class: Dictionary mapping condition labels to image paths
        """
        print(f"\n🏥 Distributing images across {self.num_hospitals} hospitals...")
        
        for condition, images in images_by_class.items():
            if not images:
                print(f"   ⚠️  No images found for class '{condition}', skipping")
                continue
            
            # Shuffle images for random distribution
            random.shuffle(images)
            
            # Split into train and test
            split_idx = int(len(images) * self.train_split)
            train_images = images[:split_idx]
            test_images = images[split_idx:]
            
            # Distribute evenly across hospitals
            for split_name, split_images in [('train', train_images), ('test', test_images)]:
                images_per_hospital = len(split_images) // self.num_hospitals
                
                for idx, hospital_id in enumerate(self.hospital_names):
                    # Calculate slice for this hospital
                    start_idx = idx * images_per_hospital
                    if idx == self.num_hospitals - 1:
                        # Last hospital gets remaining images
                        end_idx = len(split_images)
                    else:
                        end_idx = start_idx + images_per_hospital
                    
                    hospital_images = split_images[start_idx:end_idx]
                    
                    # Create directory structure
                    dest_dir = self.base_dir / f"hospital{hospital_id}" / split_name / condition
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy images
                    for img_path in hospital_images:
                        dest_path = dest_dir / img_path.name
                        shutil.copy2(img_path, dest_path)
                        
                        # Update statistics
                        self.stats[f"hospital{hospital_id}"][f"{split_name}_{condition}"] += 1
            
            print(f"   ✓ {condition}: {len(train_images)} train, {len(test_images)} test")
    
    def print_summary(self):
        """Print final distribution summary."""
        print("\n" + "="*80)
        print("📊 DISTRIBUTION SUMMARY")
        print("="*80)
        
        for hospital_id in self.hospital_names:
            hospital_key = f"hospital{hospital_id}"
            hospital_stats = self.stats[hospital_key]
            
            # Calculate totals
            train_total = sum(v for k, v in hospital_stats.items() if k.startswith('train_'))
            test_total = sum(v for k, v in hospital_stats.items() if k.startswith('test_'))
            
            print(f"\n🏥 Hospital {hospital_id}:")
            print(f"   Train: {train_total} images")
            for key, count in sorted(hospital_stats.items()):
                if key.startswith('train_'):
                    condition = key.replace('train_', '')
                    print(f"      - {condition}: {count}")
            
            print(f"   Test:  {test_total} images")
            for key, count in sorted(hospital_stats.items()):
                if key.startswith('test_'):
                    condition = key.replace('test_', '')
                    print(f"      - {condition}: {count}")
        
        print("\n" + "="*80)
        print(f"✓ Dataset organized in: {self.base_dir}")
        print("="*80)
    
    def process(self):
        """Main processing pipeline."""
        print("\n" + "="*80)
        print("MedRAG ZIP Dataset Loader")
        print("="*80)
        
        # Create temporary extraction directory
        temp_dir = self.output_dir / ".temp_extract"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Extract ZIP
            extracted_dir = self.extract_zip(temp_dir)
            
            # Step 2: Collect and classify images
            images_by_class = self.collect_images(extracted_dir)
            
            if not images_by_class or all(len(v) == 0 for v in images_by_class.values()):
                print("\n❌ Error: No valid images found in ZIP file")
                return False
            
            # Step 3: Distribute images across hospitals
            self.distribute_images(images_by_class)
            
            # Step 4: Print summary
            self.print_summary()
            
            return True
            
        finally:
            # Cleanup temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"\n🧹 Cleaned up temporary files")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract and organize X-ray ZIP datasets for federated learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python src/load_zip_dataset.py --zip-file dataset.zip --output-dir ./data
  
  # Binary classification (covid vs normal only)
  python src/load_zip_dataset.py --zip-file dataset.zip --output-dir ./data --binary
  
  # Custom train/test split
  python src/load_zip_dataset.py --zip-file dataset.zip --output-dir ./data --train-split 0.7
  
  # Different number of hospitals
  python src/load_zip_dataset.py --zip-file dataset.zip --output-dir ./data --num-hospitals 6
        """
    )
    
    parser.add_argument(
        '--zip-file',
        type=str,
        required=True,
        help='Path to ZIP file containing X-ray images'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory for organized dataset (default: ./data)'
    )
    
    parser.add_argument(
        '--num-hospitals',
        type=int,
        default=4,
        help='Number of hospitals to distribute data across (default: 4)'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Ratio of training data (default: 0.8)'
    )
    
    parser.add_argument(
        '--binary',
        action='store_true',
        help='Use only covid and normal classes (binary classification)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass  # numpy not available, skip numpy seed setting
    
    # Create loader and process
    loader = ZipDatasetLoader(
        zip_file=args.zip_file,
        output_dir=args.output_dir,
        num_hospitals=args.num_hospitals,
        train_split=args.train_split,
        binary_classification=args.binary
    )
    
    success = loader.process()
    
    if success:
        print("\n✅ Dataset preparation complete!")
        print(f"\nNext steps:")
        print(f"  1. Train with ResNet+VGG: python src/demo_rag_vfl_with_zip.py --datapath {args.output_dir}")
        print(f"  2. Train with YOLO:      python src/demo_rag_vfl_with_zip.py --datapath {args.output_dir} --model-type yolo5")
        print(f"  3. Use RAG enhancement:  python src/demo_rag_vfl_with_zip.py --datapath {args.output_dir} --use-rag")
        return 0
    else:
        print("\n❌ Dataset preparation failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
