"""
Tests for ZIP Dataset Loader

This module contains tests for:
1. ZIP dataset extraction and organization
2. Hospital naming and directory structure
3. Compatibility with VFL framework

Note: YOLO model tests have been removed as models_with_yolo.py is no longer
part of the project. The supported backbones are now resnet18, densenet121,
and efficientnet_b0 (see vfl_feature_partition.py / train_multimodel.py).
"""

import sys
import os
import unittest
import tempfile
import zipfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Try importing required modules
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Some tests will be skipped.")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Model tests will be skipped.")


class TestZipDatasetLoader(unittest.TestCase):
    """Test the ZIP dataset loader functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        if not PIL_AVAILABLE:
            raise unittest.SkipTest("PIL not available")
        
        # Create temporary directory for tests
        cls.temp_dir = tempfile.mkdtemp()
        cls.zip_path = Path(cls.temp_dir) / "test_dataset.zip"
        cls.output_dir = Path(cls.temp_dir) / "output"
        
        # Create a sample ZIP file with images
        cls._create_sample_zip()
    
    @classmethod
    def _create_sample_zip(cls):
        """Create a sample ZIP file with test images."""
        # Create temporary directory for images
        img_dir = Path(cls.temp_dir) / "images"
        img_dir.mkdir(exist_ok=True)
        
        # Create sample folders
        covid_dir = img_dir / "covid"
        normal_dir = img_dir / "normal"
        covid_dir.mkdir(exist_ok=True)
        normal_dir.mkdir(exist_ok=True)
        
        # Create sample images (simple colored squares)
        for i in range(10):
            # COVID images (red)
            img = Image.new('RGB', (224, 224), color='red')
            img.save(covid_dir / f"covid_{i:03d}.jpg")
            
            # Normal images (green)
            img = Image.new('RGB', (224, 224), color='green')
            img.save(normal_dir / f"normal_{i:03d}.jpg")
        
        # Create ZIP file
        with zipfile.ZipFile(cls.zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(img_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(img_dir)
                    zipf.write(file_path, arcname)
        
        # Clean up image directory
        shutil.rmtree(img_dir)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    def test_import_loader(self):
        """Test that the ZIP dataset loader can be imported."""
        try:
            from load_zip_dataset import ZipDatasetLoader
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Could not import ZipDatasetLoader: {e}")
    
    def test_loader_initialization(self):
        """Test that the loader can be initialized."""
        from load_zip_dataset import ZipDatasetLoader
        
        loader = ZipDatasetLoader(
            zip_file=str(self.zip_path),
            output_dir=str(self.output_dir),
            num_hospitals=4,
            train_split=0.8
        )
        
        self.assertEqual(loader.num_hospitals, 4)
        self.assertEqual(loader.train_split, 0.8)
        self.assertEqual(len(loader.hospital_names), 4)
        self.assertEqual(loader.hospital_names, ['A', 'B', 'C', 'D'])
    
    def test_zip_extraction_and_organization(self):
        """Test that ZIP extraction and organization works correctly."""
        from load_zip_dataset import ZipDatasetLoader
        
        loader = ZipDatasetLoader(
            zip_file=str(self.zip_path),
            output_dir=str(self.output_dir),
            num_hospitals=4,
            train_split=0.8,
            binary_classification=True
        )
        
        # Process the dataset
        success = loader.process()
        self.assertTrue(success, "Dataset processing should succeed")
        
        # Check that hospital directories were created
        base_dir = self.output_dir / "SplitCovid19"
        self.assertTrue(base_dir.exists(), "Base directory should exist")
        
        for hospital_id in ['A', 'B', 'C', 'D']:
            hospital_dir = base_dir / f"hospital{hospital_id}"
            self.assertTrue(hospital_dir.exists(), f"Hospital {hospital_id} directory should exist")
            
            # Check train and test directories
            train_dir = hospital_dir / "train"
            test_dir = hospital_dir / "test"
            self.assertTrue(train_dir.exists(), f"Hospital {hospital_id} train directory should exist")
            self.assertTrue(test_dir.exists(), f"Hospital {hospital_id} test directory should exist")
            
            # Check class directories
            for class_name in ['covid', 'normal']:
                train_class_dir = train_dir / class_name
                test_class_dir = test_dir / class_name
                self.assertTrue(train_class_dir.exists(), 
                              f"Hospital {hospital_id} train/{class_name} should exist")
                self.assertTrue(test_class_dir.exists(), 
                              f"Hospital {hospital_id} test/{class_name} should exist")
    
    def test_image_distribution(self):
        """Test that images are distributed across hospitals."""
        from load_zip_dataset import ZipDatasetLoader
        
        loader = ZipDatasetLoader(
            zip_file=str(self.zip_path),
            output_dir=str(self.output_dir),
            num_hospitals=4,
            train_split=0.8,
            binary_classification=True
        )
        
        loader.process()
        
        # Count images in each hospital
        base_dir = self.output_dir / "SplitCovid19"
        total_images = 0
        
        for hospital_id in ['A', 'B', 'C', 'D']:
            hospital_dir = base_dir / f"hospital{hospital_id}"
            
            # Count train and test images
            train_covid = list((hospital_dir / "train" / "covid").glob("*.jpg"))
            train_normal = list((hospital_dir / "train" / "normal").glob("*.jpg"))
            test_covid = list((hospital_dir / "test" / "covid").glob("*.jpg"))
            test_normal = list((hospital_dir / "test" / "normal").glob("*.jpg"))
            
            hospital_total = len(train_covid) + len(train_normal) + len(test_covid) + len(test_normal)
            total_images += hospital_total
            
            # Each hospital should have some images
            self.assertGreater(hospital_total, 0, f"Hospital {hospital_id} should have images")
        
        # Total should be 20 (10 covid + 10 normal)
        self.assertEqual(total_images, 20, "Total images should be 20")


class TestHospitalNaming(unittest.TestCase):
    """Test that hospital naming is used correctly."""
    
    def test_hospital_naming_in_loader(self):
        """Test that ZIP loader uses hospital naming."""
        if not PIL_AVAILABLE:
            self.skipTest("PIL not available")
        
        from load_zip_dataset import ZipDatasetLoader
        
        loader = ZipDatasetLoader(
            zip_file="dummy.zip",
            output_dir="dummy_output",
            num_hospitals=4
        )
        
        self.assertEqual(loader.hospital_names, ['A', 'B', 'C', 'D'])
    
    def test_hospital_naming_with_6_hospitals(self):
        """Test that loader supports different number of hospitals."""
        if not PIL_AVAILABLE:
            self.skipTest("PIL not available")
        
        from load_zip_dataset import ZipDatasetLoader
        
        loader = ZipDatasetLoader(
            zip_file="dummy.zip",
            output_dir="dummy_output",
            num_hospitals=6
        )
        
        self.assertEqual(loader.hospital_names, ['A', 'B', 'C', 'D', 'E', 'F'])


def run_tests():
    """Run all tests."""
    print("="*80)
    print("Running Tests for ZIP Dataset Loader")
    print("="*80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestZipDatasetLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestHospitalNaming))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
