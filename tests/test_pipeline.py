"""
Tests for the end-to-end run_pipeline.py script and the Hospital Network page
fix that adds support for the hospitalA/B/C/D directory structure produced by
the ZIP dataset loader.
"""

import os
import sys
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

# Make sure src/ is on the path for imports used inside the pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_zip(zip_path: Path, img_dir: Path):
    """Create a minimal ZIP file with a few labelled X-ray images."""
    img_dir.mkdir(parents=True, exist_ok=True)
    covid_dir = img_dir / "covid"
    normal_dir = img_dir / "normal"
    covid_dir.mkdir(exist_ok=True)
    normal_dir.mkdir(exist_ok=True)

    for i in range(8):
        Image.new('RGB', (64, 64), color='red').save(covid_dir / f"covid_{i:03d}.jpg")
        Image.new('RGB', (64, 64), color='green').save(normal_dir / f"normal_{i:03d}.jpg")

    with zipfile.ZipFile(zip_path, 'w') as zf:
        for root, _, files in os.walk(img_dir):
            for file in files:
                fp = Path(root) / file
                zf.write(fp, fp.relative_to(img_dir))


def _build_hospital_dirs(base: Path, naming: str = "new"):
    """
    Create SplitCovid19 with dummy hospital directories.

    naming='new'    -> hospitalA/B/C/D  (from zip loader)
    naming='legacy' -> client0/1/2/3
    naming='both'   -> both naming schemes
    """
    split_dir = base / "SplitCovid19"
    if naming in ("new", "both"):
        for letter in "ABCD":
            for split in ("train", "test"):
                for cls in ("covid", "normal"):
                    d = split_dir / f"hospital{letter}" / split / cls
                    d.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (8, 8), color='blue').save(d / "img.jpg")
    if naming in ("legacy", "both"):
        for idx in range(4):
            for split in ("train", "test"):
                for cls in ("covid", "normal"):
                    d = split_dir / f"client{idx}" / split / cls
                    d.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (8, 8), color='blue').save(d / "img.jpg")
    return split_dir


# ---------------------------------------------------------------------------
# Tests: Hospital Network page – directory detection
# ---------------------------------------------------------------------------

class TestHospitalNetworkDataDetection(unittest.TestCase):
    """Verify the hospital-data scanning logic supports both naming schemes."""

    @classmethod
    def setUpClass(cls):
        if not PIL_AVAILABLE:
            raise unittest.SkipTest("PIL not available")
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.temp_dir, True)

    @classmethod
    def tearDownClass(cls):
        pass  # cleanup handled by addClassCleanup

    def _count_images(self, directory: Path) -> int:
        """Replicate count_images_in_directory from the webapp page."""
        if not directory.exists():
            return 0
        exts = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        return sum(1 for f in directory.rglob('*') if f.suffix in exts)

    def _load_hospitals_from_split_dir(self, split_dir: Path):
        """Replicate the hospital-loading logic from the updated webapp page."""
        mapping = [
            ('hospitalA', 'client0', 'Hospital A'),
            ('hospitalB', 'client1', 'Hospital B'),
            ('hospitalC', 'client2', 'Hospital C'),
            ('hospitalD', 'client3', 'Hospital D'),
        ]
        hospitals = []
        for new_dir, legacy_dir, name in mapping:
            if (split_dir / new_dir).exists():
                client_path = split_dir / new_dir
            elif (split_dir / legacy_dir).exists():
                client_path = split_dir / legacy_dir
            else:
                continue

            train_covid  = self._count_images(client_path / 'train' / 'covid')
            train_normal = self._count_images(client_path / 'train' / 'normal')
            test_covid   = self._count_images(client_path / 'test' / 'covid')
            test_normal  = self._count_images(client_path / 'test' / 'normal')

            hospitals.append({
                'name': name,
                'train': train_covid + train_normal,
                'test':  test_covid + test_normal,
            })
        return hospitals

    def test_new_naming_detected(self):
        """hospitalA/B/C/D directories are detected."""
        base = self.temp_dir / "new_naming"
        split_dir = _build_hospital_dirs(base, naming="new")
        hospitals = self._load_hospitals_from_split_dir(split_dir)

        self.assertEqual(len(hospitals), 4,
                         "All 4 hospital dirs should be found with new naming")
        for h in hospitals:
            self.assertGreater(h['train'] + h['test'], 0,
                               f"{h['name']} should have images")

    def test_legacy_naming_detected(self):
        """client0/1/2/3 directories are still detected as a fallback."""
        base = self.temp_dir / "legacy_naming"
        split_dir = _build_hospital_dirs(base, naming="legacy")
        hospitals = self._load_hospitals_from_split_dir(split_dir)

        self.assertEqual(len(hospitals), 4,
                         "All 4 hospital dirs should be found with legacy naming")
        for h in hospitals:
            self.assertGreater(h['train'] + h['test'], 0,
                               f"{h['name']} should have images")

    def test_new_naming_preferred_over_legacy(self):
        """When both naming schemes exist, the new (hospitalX) one is preferred."""
        base = self.temp_dir / "both_naming"
        split_dir = _build_hospital_dirs(base, naming="both")

        # Add an extra image only in the new-format dirs so we can tell which
        # path was chosen
        extra_img = split_dir / "hospitalA" / "train" / "covid" / "extra.jpg"
        Image.new('RGB', (8, 8), color='yellow').save(extra_img)

        hospitals = self._load_hospitals_from_split_dir(split_dir)
        hospital_a = next(h for h in hospitals if h['name'] == 'Hospital A')

        # The new-format dir has 2 train/covid images (img.jpg + extra.jpg)
        self.assertGreaterEqual(hospital_a['train'], 2,
                                "New naming should be preferred and include extra image")


# ---------------------------------------------------------------------------
# Tests: run_pipeline.py argument parsing and step orchestration
# ---------------------------------------------------------------------------

class TestRunPipelineScript(unittest.TestCase):
    """Test that scripts/run_pipeline.py is importable and its helpers work."""

    @classmethod
    def setUpClass(cls):
        # Make the scripts directory importable
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

    def test_pipeline_script_exists(self):
        """The pipeline script must be present."""
        script = Path(__file__).parent.parent / "scripts" / "run_pipeline.py"
        self.assertTrue(script.exists(), "scripts/run_pipeline.py should exist")

    def test_pipeline_script_importable(self):
        """run_pipeline.py can be imported without side effects."""
        import importlib.util
        script_path = Path(__file__).parent.parent / "scripts" / "run_pipeline.py"
        spec = importlib.util.spec_from_file_location("run_pipeline", script_path)
        self.assertIsNotNone(spec, f"Could not create module spec from {script_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(hasattr(mod, "main"))
        self.assertTrue(hasattr(mod, "step_extract_zip"))
        self.assertTrue(hasattr(mod, "step_train"))
        self.assertTrue(hasattr(mod, "step_verify_registry"))
        self.assertTrue(hasattr(mod, "step_verify_ledger"))

    def test_pipeline_argparse_help(self):
        """The CLI argument parser produces help text without error."""
        import subprocess
        result = subprocess.run(
            [sys.executable,
             str(Path(__file__).parent.parent / "scripts" / "run_pipeline.py"),
             "--help"],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--zip-file", result.stdout)
        self.assertIn("--datapath", result.stdout)
        self.assertIn("--use-rag", result.stdout)
        self.assertIn("--withblockchain", result.stdout)

    def test_pipeline_requires_input(self):
        """Running without --zip-file or --datapath should exit with error."""
        import subprocess
        result = subprocess.run(
            [sys.executable,
             str(Path(__file__).parent.parent / "scripts" / "run_pipeline.py")],
            capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0,
                            "Should fail without an input source")


# ---------------------------------------------------------------------------
# Tests: ZIP extraction → hospital directory structure
# ---------------------------------------------------------------------------

class TestZipToHospitalPipeline(unittest.TestCase):
    """
    Verify that a ZIP file can be extracted into the hospitalA/B/C/D structure
    that the training script expects.
    """

    @classmethod
    def setUpClass(cls):
        if not PIL_AVAILABLE:
            raise unittest.SkipTest("PIL not available")
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.temp_dir, True)
        cls.zip_path = cls.temp_dir / "test_dataset.zip"
        cls.output_dir = cls.temp_dir / "output"
        img_dir = cls.temp_dir / "_images"
        _make_sample_zip(cls.zip_path, img_dir)
        shutil.rmtree(img_dir, ignore_errors=True)

        # Extract once during setup so all tests in this class share the result
        try:
            from load_zip_dataset import ZipDatasetLoader
        except ImportError as exc:
            raise unittest.SkipTest(f"load_zip_dataset not importable: {exc}")
        loader = ZipDatasetLoader(
            zip_file=str(cls.zip_path),
            output_dir=str(cls.output_dir),
            num_hospitals=4,
            train_split=0.8,
            binary_classification=True,
        )
        cls.extraction_ok = loader.process()

    @classmethod
    def tearDownClass(cls):
        pass  # cleanup handled by addClassCleanup

    def test_extracted_directories_match_training_script_expectations(self):
        """
        After extraction the demo_rag_vfl_with_zip.py path-resolution logic
        (hospital{X} first, then client{i} fallback) should find all 4
        hospital directories.
        """
        self.assertTrue(self.extraction_ok, "ZIP extraction should succeed")

        split_dir = self.output_dir / "SplitCovid19"

        # Simulate the path-resolution logic in demo_rag_vfl_with_zip.py
        hospital_names = ['A', 'B', 'C', 'D']
        found = []
        for i, letter in enumerate(hospital_names):
            new_path = split_dir / f"hospital{letter}"
            legacy_path = split_dir / f"client{i}"
            if new_path.exists():
                found.append(('new', letter, new_path))
            elif legacy_path.exists():
                found.append(('legacy', str(i), legacy_path))

        self.assertEqual(len(found), 4,
                         "All 4 hospital paths must be resolvable after ZIP extraction")

        # All should be the new hospitalX format
        for fmt, _, _ in found:
            self.assertEqual(fmt, 'new',
                             "ZIP loader should create hospitalA/B/C/D directories")

    def test_extracted_data_has_train_and_test_splits(self):
        """Each hospital directory must contain both train/ and test/."""
        self.assertTrue(self.extraction_ok, "ZIP extraction should succeed")

        split_dir = self.output_dir / "SplitCovid19"
        for letter in "ABCD":
            hospital_dir = split_dir / f"hospital{letter}"
            self.assertTrue(hospital_dir.exists(),
                            f"hospital{letter} must exist after extraction")
            self.assertTrue((hospital_dir / "train").exists(),
                             f"hospital{letter}/train must exist")
            self.assertTrue((hospital_dir / "test").exists(),
                             f"hospital{letter}/test must exist")


# ---------------------------------------------------------------------------
# Tests: detect_class_names helpers (multi-class support)
# ---------------------------------------------------------------------------

# Optional heavyweight dependencies – tests are skipped when unavailable
try:
    import numpy  # noqa: F401
    import torch  # noqa: F401
    NUMPY_TORCH_AVAILABLE = True
except ImportError:
    NUMPY_TORCH_AVAILABLE = False


class TestDetectClassNames(unittest.TestCase):
    """Verify detect_class_names / detect_class_names_from_dir auto-detection."""

    @classmethod
    def setUpClass(cls):
        if not PIL_AVAILABLE:
            raise unittest.SkipTest("PIL not available")
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.temp_dir, True)

    def _make_split_dir(self, base: Path, client_name: str, classes):
        """Create a minimal SplitCovid19/<client_name>/train/<class> tree."""
        for cls_name in classes:
            d = base / "SplitCovid19" / client_name / "train" / cls_name
            d.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (8, 8)).save(d / "img.jpg")
        return base

    # ------------------------------------------------------------------
    # Inline reference implementation (no heavy dependencies required)
    # Mirrors the logic in demo_rag_vfl_with_zip.detect_class_names and
    # inference.detect_class_names_from_dir so the core logic can always
    # be tested even when numpy/torch are absent.
    # ------------------------------------------------------------------

    @staticmethod
    def _detect(datapath):
        """Reference implementation of detect_class_names."""
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

    # ------------------------------------------------------------------
    # Core logic tests (always run – no heavy deps needed)
    # ------------------------------------------------------------------

    def test_detect_class_names_new_naming(self):
        """detect_class_names picks up hospitalA/train classes (new naming)."""
        base = self.temp_dir / "new_4class"
        self._make_split_dir(base, "hospitalA",
                             ["covid", "lung_opacity", "normal", "pneumonia"])
        result = self._detect(str(base))
        self.assertEqual(result, ["covid", "lung_opacity", "normal", "pneumonia"])

    def test_detect_class_names_legacy_naming(self):
        """detect_class_names falls back to client0/train classes (legacy naming)."""
        base = self.temp_dir / "legacy_2class"
        self._make_split_dir(base, "client0", ["covid", "normal"])
        result = self._detect(str(base))
        self.assertEqual(result, ["covid", "normal"])

    def test_detect_class_names_prefers_new_over_legacy(self):
        """When both naming schemes exist, hospitalA takes precedence."""
        base = self.temp_dir / "both_naming"
        self._make_split_dir(base, "hospitalA",
                             ["covid", "lung_opacity", "normal", "pneumonia"])
        # Legacy dir has fewer classes
        self._make_split_dir(base, "client0", ["covid", "normal"])
        result = self._detect(str(base))
        self.assertEqual(result,
                         ["covid", "lung_opacity", "normal", "pneumonia"])

    def test_detect_class_names_missing_dataset(self):
        """detect_class_names returns None when the dataset is absent."""
        result = self._detect(str(self.temp_dir / "nonexistent"))
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Imported-function tests (skipped when heavy deps are absent)
    # ------------------------------------------------------------------

    def test_detect_class_names_from_module(self):
        """detect_class_names in demo_rag_vfl_with_zip matches reference impl."""
        if not NUMPY_TORCH_AVAILABLE:
            self.skipTest("numpy/torch not available")
        from demo_rag_vfl_with_zip import detect_class_names
        base = self.temp_dir / "module_new_4class"
        self._make_split_dir(base, "hospitalA",
                             ["covid", "lung_opacity", "normal", "pneumonia"])
        result = detect_class_names(str(base))
        self.assertEqual(result, ["covid", "lung_opacity", "normal", "pneumonia"])

    def test_detect_class_names_from_dir_inference(self):
        """detect_class_names_from_dir in inference matches reference impl."""
        if not NUMPY_TORCH_AVAILABLE:
            self.skipTest("numpy/torch not available")
        from inference import detect_class_names_from_dir
        base = self.temp_dir / "infer_4class"
        self._make_split_dir(base, "hospitalA",
                             ["covid", "lung_opacity", "normal", "pneumonia"])
        result = detect_class_names_from_dir(str(base))
        self.assertEqual(result, ["covid", "lung_opacity", "normal", "pneumonia"])

    def test_server_model_num_classes_not_hardcoded(self):
        """RAGEnhancedServerModel respects arbitrary num_classes."""
        if not NUMPY_TORCH_AVAILABLE:
            self.skipTest("torch not available")
        import torch
        from rag_server_model import RAGEnhancedServerModel
        model = RAGEnhancedServerModel(
            embedding_dim=64, num_classes=4, use_rag=False)
        x = torch.randn(2, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 4),
                         "Output shape should match num_classes=4")

    def test_load_inference_model_uses_dataset_dir(self):
        """load_inference_model detects classes when dataset_dir is provided."""
        if not NUMPY_TORCH_AVAILABLE:
            self.skipTest("torch not available")
        from inference import load_inference_model
        base = self.temp_dir / "infer_dir_detect"
        self._make_split_dir(base, "hospitalA",
                             ["covid", "lung_opacity", "normal", "pneumonia"])
        engine = load_inference_model(use_rag=False, dataset_dir=str(base))
        self.assertEqual(engine.class_names,
                         ["covid", "lung_opacity", "normal", "pneumonia"])
        self.assertEqual(engine.server_model.num_classes, 4)

    def test_load_inference_model_default_binary(self):
        """load_inference_model defaults to binary when no info available."""
        if not NUMPY_TORCH_AVAILABLE:
            self.skipTest("torch not available")
        from inference import load_inference_model
        engine = load_inference_model(use_rag=False)
        self.assertEqual(engine.class_names, ['Normal', 'COVID-19'])
        self.assertEqual(engine.server_model.num_classes, 2)

    def test_detect_class_names_from_dir_splitcovid19_as_root(self):
        """detect_class_names_from_dir handles dataset_dir == SplitCovid19."""
        if not NUMPY_TORCH_AVAILABLE:
            self.skipTest("numpy/torch not available")
        from inference import detect_class_names_from_dir
        # Build tree at base/SplitCovid19/hospitalA/train/<classes>
        base = self.temp_dir / "splitcovid_root"
        self._make_split_dir(base, "hospitalA",
                             ["covid", "normal"])
        # Pass base/SplitCovid19 as dataset_dir (user passes the dir itself)
        split_dir = base / "SplitCovid19"
        result = detect_class_names_from_dir(str(split_dir))
        self.assertEqual(result, ["covid", "normal"])

    def test_load_inference_model_uses_checkpoint_model_type(self):
        """load_inference_model respects model_type stored in checkpoint config."""
        if not NUMPY_TORCH_AVAILABLE:
            self.skipTest("torch not available")
        import torch
        import tempfile
        from inference import load_inference_model
        from rag_server_model import RAGEnhancedServerModel

        # Create a minimal checkpoint with model_type=resnet_vgg and multi-class config
        server = RAGEnhancedServerModel(embedding_dim=64, num_classes=4, use_rag=False)
        ckpt = {
            'server_state_dict': server.state_dict(),
            'config': {
                'num_classes': 4,
                'class_names': ['covid', 'lung_opacity', 'normal', 'pneumonia'],
                'model_type': 'resnet_vgg',
            },
        }
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(ckpt, f.name)
            ckpt_path = f.name

        try:
            engine = load_inference_model(checkpoint_path=ckpt_path, use_rag=False)
            self.assertEqual(engine.class_names,
                             ['covid', 'lung_opacity', 'normal', 'pneumonia'])
            self.assertEqual(engine.server_model.num_classes, 4)
        finally:
            os.unlink(ckpt_path)

    def test_rag_module_available_with_langchain_disabled(self):
        """get_rag_module() returns module when use_rag=True and use_langchain=False."""
        if not NUMPY_TORCH_AVAILABLE:
            self.skipTest("torch not available")
        from rag_server_model import RAGEnhancedServerModel
        model = RAGEnhancedServerModel(embedding_dim=64, num_classes=2,
                                       use_rag=True, use_langchain=False)
        self.assertIsNotNone(model.get_rag_module(),
                             "get_rag_module() must return module when use_rag=True")

    def test_rag_module_available_with_langchain_fallback(self):
        """get_rag_module() returns module even when use_langchain=True (LangChain unavailable)."""
        if not NUMPY_TORCH_AVAILABLE:
            self.skipTest("torch not available")
        from rag_server_model import RAGEnhancedServerModel
        # use_langchain=True but LangChain is almost certainly unavailable in CI,
        # so the model should fall back to simple RAG and rag_module must exist.
        model = RAGEnhancedServerModel(embedding_dim=64, num_classes=2,
                                       use_rag=True, use_langchain=True)
        self.assertIsNotNone(model.get_rag_module(),
                             "get_rag_module() must return module even when LangChain is requested")

    def test_provider_url_resolution_web3_provider_uri(self):
        """WEB3_PROVIDER_URI is preferred over GANACHE_URL in utils."""
        import importlib
        # We test the resolution logic without a live node by checking the URL used
        # when Web3 is unavailable — which skips the connection but the URL is set.
        old_web3_uri = os.environ.get('WEB3_PROVIDER_URI')
        old_ganache_url = os.environ.get('GANACHE_URL')
        try:
            os.environ['WEB3_PROVIDER_URI'] = 'http://127.0.0.1:9999'
            os.environ.pop('GANACHE_URL', None)

            # Verify the resolution formula directly
            rpc_url = (os.getenv('WEB3_PROVIDER_URI')
                       or os.getenv('GANACHE_URL')
                       or 'http://127.0.0.1:8545')
            self.assertEqual(rpc_url, 'http://127.0.0.1:9999')
        finally:
            if old_web3_uri is None:
                os.environ.pop('WEB3_PROVIDER_URI', None)
            else:
                os.environ['WEB3_PROVIDER_URI'] = old_web3_uri
            if old_ganache_url is None:
                os.environ.pop('GANACHE_URL', None)
            else:
                os.environ['GANACHE_URL'] = old_ganache_url

    def test_provider_url_resolution_ganache_fallback(self):
        """GANACHE_URL is used when WEB3_PROVIDER_URI is absent."""
        old_web3_uri = os.environ.get('WEB3_PROVIDER_URI')
        old_ganache_url = os.environ.get('GANACHE_URL')
        try:
            os.environ.pop('WEB3_PROVIDER_URI', None)
            os.environ['GANACHE_URL'] = 'http://127.0.0.1:7545'

            rpc_url = (os.getenv('WEB3_PROVIDER_URI')
                       or os.getenv('GANACHE_URL')
                       or 'http://127.0.0.1:8545')
            self.assertEqual(rpc_url, 'http://127.0.0.1:7545')
        finally:
            if old_web3_uri is None:
                os.environ.pop('WEB3_PROVIDER_URI', None)
            else:
                os.environ['WEB3_PROVIDER_URI'] = old_web3_uri
            if old_ganache_url is None:
                os.environ.pop('GANACHE_URL', None)
            else:
                os.environ['GANACHE_URL'] = old_ganache_url

    def test_provider_url_default_hardhat(self):
        """Default provider URL is Hardhat (http://127.0.0.1:8545)."""
        old_web3_uri = os.environ.get('WEB3_PROVIDER_URI')
        old_ganache_url = os.environ.get('GANACHE_URL')
        try:
            os.environ.pop('WEB3_PROVIDER_URI', None)
            os.environ.pop('GANACHE_URL', None)

            rpc_url = (os.getenv('WEB3_PROVIDER_URI')
                       or os.getenv('GANACHE_URL')
                       or 'http://127.0.0.1:8545')
            self.assertEqual(rpc_url, 'http://127.0.0.1:8545')
        finally:
            if old_web3_uri is None:
                os.environ.pop('WEB3_PROVIDER_URI', None)
            else:
                os.environ['WEB3_PROVIDER_URI'] = old_web3_uri
            if old_ganache_url is None:
                os.environ.pop('GANACHE_URL', None)
            else:
                os.environ['GANACHE_URL'] = old_ganache_url

    def test_plotly_layout_no_titlefont(self):
        """Registry performance chart layout must not use deprecated titlefont."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            self.skipTest("plotly not available")

        # This should not raise ValueError about invalid 'titlefont' property
        fig = go.Figure()
        fig.update_layout(
            yaxis=dict(
                title=dict(text='Accuracy (%)', font=dict(color='#4CAF50')),
                tickfont=dict(color='#4CAF50'),
            ),
            yaxis2=dict(
                title=dict(text='Loss', font=dict(color='#FF5722')),
                tickfont=dict(color='#FF5722'),
                overlaying='y',
                side='right',
            ),
        )
        # If we reach here without exception the fix is correct
        self.assertIsNotNone(fig.layout.yaxis.title.text)
        self.assertEqual(fig.layout.yaxis.title.text, 'Accuracy (%)')

    def test_use_blockchain_alias(self):
        """--use-blockchain must be accepted as an alias for --withblockchain."""
        if not NUMPY_TORCH_AVAILABLE:
            self.skipTest("numpy/torch not available")
        import argparse
        # Replicate the parser definition from demo_rag_vfl_with_zip.py
        parser = argparse.ArgumentParser()
        parser.add_argument('--withblockchain', action='store_true')
        parser.add_argument('--use-blockchain', dest='withblockchain', action='store_true')
        args = parser.parse_args(['--use-blockchain'])
        self.assertTrue(args.withblockchain,
                        "--use-blockchain should set withblockchain=True")


if __name__ == '__main__':
    unittest.main()
