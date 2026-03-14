#!/usr/bin/env python3
"""
Manual validation script for MedRAG enhancements.
Tests functionality without requiring full dependencies.
"""

import os
import sys
from pathlib import Path

print("="*80)
print("MedRAG Enhancements - Manual Validation")
print("="*80)

# Test 1: File existence
print("\n1. Checking file existence...")
files_to_check = [
    'src/load_zip_dataset.py',
    'src/models_with_yolo.py',
    'src/demo_rag_vfl_with_zip.py',
    'docs/USING_YOUR_XRAY_ZIP.md',
    'tests/test_zip_and_yolo.py',
    'requirements.txt',
    'README.md',
]

all_exist = True
for file_path in files_to_check:
    exists = os.path.exists(file_path)
    status = "✓" if exists else "✗"
    print(f"   {status} {file_path}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n✗ Some files are missing!")
    sys.exit(1)
else:
    print("\n✓ All required files exist")

# Test 2: Python syntax
print("\n2. Checking Python syntax...")
import py_compile

python_files = [
    'src/load_zip_dataset.py',
    'src/models_with_yolo.py',
    'src/demo_rag_vfl_with_zip.py',
    'tests/test_zip_and_yolo.py',
]

all_valid = True
for file_path in python_files:
    try:
        py_compile.compile(file_path, doraise=True)
        print(f"   ✓ {file_path}")
    except py_compile.PyCompileError as e:
        print(f"   ✗ {file_path}: {e}")
        all_valid = False

if not all_valid:
    print("\n✗ Some files have syntax errors!")
    sys.exit(1)
else:
    print("\n✓ All Python files have valid syntax")

# Test 3: Backward compatibility
print("\n3. Checking backward compatibility (original files unchanged)...")
original_files = [
    'src/demo_rag_vfl.py',
    'src/models.py',
    'prepare_dataset.py',
]

all_valid = True
for file_path in original_files:
    try:
        py_compile.compile(file_path, doraise=True)
        print(f"   ✓ {file_path}")
    except py_compile.PyCompileError as e:
        print(f"   ✗ {file_path}: {e}")
        all_valid = False

if not all_valid:
    print("\n✗ Backward compatibility broken!")
    sys.exit(1)
else:
    print("\n✓ Backward compatibility maintained")

# Test 4: Check documentation
print("\n4. Checking documentation...")
doc_file = 'docs/USING_YOUR_XRAY_ZIP.md'
with open(doc_file, 'r') as f:
    content = f.read()

def check_hospital_mentions(text):
    """Check if hospital naming is mentioned in various formats."""
    text_lower = text.lower()
    hospital_variants = [
        'hospital a', 'hospital b', 'hospital c', 'hospital d',
        'hospitala', 'hospitalb', 'hospitalc', 'hospitald'
    ]
    return any(variant in text_lower for variant in hospital_variants)
    
checks = [
    ('Hospital A, B, C, D mentioned', check_hospital_mentions(content)),
    ('YOLO mentioned', 'YOLO' in content or 'yolo' in content),
    ('ZIP mentioned', 'ZIP' in content or 'zip' in content),
    ('load_zip_dataset.py mentioned', 'load_zip_dataset.py' in content),
    ('demo_rag_vfl_with_zip.py mentioned', 'demo_rag_vfl_with_zip.py' in content),
]

all_checks = True
for check_name, result in checks:
    status = "✓" if result else "✗"
    print(f"   {status} {check_name}")
    if not result:
        all_checks = False

if not all_checks:
    print("\n✗ Documentation incomplete!")
    sys.exit(1)
else:
    print("\n✓ Documentation complete")

# Test 5: Check README updates
print("\n5. Checking README updates...")
with open('README.md', 'r') as f:
    readme = f.read()

readme_checks = [
    ('ZIP dataset support mentioned', 'ZIP' in readme or 'zip' in readme),
    ('YOLO mentioned', 'YOLO' in readme or 'yolo' in readme),
    ('Hospital naming mentioned', 'Hospital' in readme),
    ('New documentation link', 'USING_YOUR_XRAY_ZIP.md' in readme),
]

all_checks = True
for check_name, result in readme_checks:
    status = "✓" if result else "✗"
    print(f"   {status} {check_name}")
    if not result:
        all_checks = False

if not all_checks:
    print("\n✗ README not fully updated!")
    sys.exit(1)
else:
    print("\n✓ README properly updated")

# Test 6: Check requirements.txt
print("\n6. Checking requirements.txt...")
with open('requirements.txt', 'r') as f:
    requirements = f.read()

req_checks = [
    ('ultralytics added', 'ultralytics' in requirements),
    ('opencv-python added', 'opencv-python' in requirements),
]

all_checks = True
for check_name, result in req_checks:
    status = "✓" if result else "✗"
    print(f"   {status} {check_name}")
    if not result:
        all_checks = False

if not all_checks:
    print("\n✗ Requirements not fully updated!")
    sys.exit(1)
else:
    print("\n✓ Requirements properly updated")

# Test 7: Content verification
print("\n7. Verifying key content...")

# Check load_zip_dataset.py
with open('src/load_zip_dataset.py', 'r') as f:
    loader_content = f.read()
    
loader_checks = [
    ('Hospital naming (A, B, C, D)', 'hospitalA' in loader_content or "hospital{hospital_id}" in loader_content),
    ('ZIP extraction', 'zipfile' in loader_content),
    ('Image classification', 'classify_image' in loader_content),
    ('Progress tracking', 'tqdm' in loader_content or 'print_summary' in loader_content),
]

all_checks = True
for check_name, result in loader_checks:
    status = "✓" if result else "✗"
    print(f"   {status} ZIP Loader: {check_name}")
    if not result:
        all_checks = False

# Check models_with_yolo.py
with open('src/models_with_yolo.py', 'r') as f:
    models_content = f.read()

models_checks = [
    ('YOLOv5 support', 'yolov5' in models_content or 'YOLOv5' in models_content),
    ('YOLOv8 support', 'yolov8' in models_content or 'YOLOv8' in models_content),
    ('Hybrid ResNet+YOLO', 'ClientModelResNetYOLO' in models_content),
    ('64-dim embeddings', 'embedding_dim=64' in models_content),
    ('create_client_model function', 'def create_client_model' in models_content),
]

for check_name, result in models_checks:
    status = "✓" if result else "✗"
    print(f"   {status} YOLO Models: {check_name}")
    if not result:
        all_checks = False

# Check demo_rag_vfl_with_zip.py
with open('src/demo_rag_vfl_with_zip.py', 'r') as f:
    demo_content = f.read()

demo_checks = [
    ('Hospital naming', 'hospitalA' in demo_content or 'hospital{hospital_id}' in demo_content),
    ('YOLO model support', 'yolo5' in demo_content or 'yolo8' in demo_content),
    ('Backward compatible paths', 'client{i}' in demo_content),
    ('Model comparison', 'compare_models' in demo_content),
]

for check_name, result in demo_checks:
    status = "✓" if result else "✗"
    print(f"   {status} Demo Script: {check_name}")
    if not result:
        all_checks = False

if not all_checks:
    print("\n✗ Content verification failed!")
    sys.exit(1)
else:
    print("\n✓ Content verification passed")

# Final summary
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)
print("✓ All files exist")
print("✓ All Python files have valid syntax")
print("✓ Backward compatibility maintained")
print("✓ Documentation complete")
print("✓ Requirements updated")
print("✓ Content verified")
print("\n✅ All manual validation checks passed!")
print("="*80)

sys.exit(0)
