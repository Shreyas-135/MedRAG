"""
Tests for Cryptographic Provenance Anchoring

Tests:
- Bundle canonicalization and deterministic hashing
- Individual hash helper functions
- Signature verification (with a known key pair)
- Bundle integrity verification (verify_bundle)
- ProvenanceIntegrator mock mode (no Ganache required)
"""

import sys
import os
import json
import hashlib
import time
import unittest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from provenance import (
    build_provenance_bundle,
    compute_bundle_hash,
    hash_prompt,
    hash_generation_params,
    hash_retrieval_params,
    hash_model_version,
    verify_signature,
    verify_bundle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DUMMY_HASH = "a" * 64  # valid 64-char hex placeholder


def _make_bundle(**overrides):
    """Create a minimal valid provenance bundle."""
    kwargs = dict(
        knowledge_base_hash=_DUMMY_HASH,
        explanation_hash=_DUMMY_HASH,
        retrieval_hash=_DUMMY_HASH,
        model_version_hash=_DUMMY_HASH,
        prompt_hash=_DUMMY_HASH,
        generation_params_hash=_DUMMY_HASH,
        hospital_id="hospital-1",
        site_id="site-A",
        device_id="device-X",
        bundle_version="1.0",
        timestamp=1_700_000_000.0,
    )
    kwargs.update(overrides)
    return build_provenance_bundle(**kwargs)


# ---------------------------------------------------------------------------
# Hash helper tests
# ---------------------------------------------------------------------------

class TestHashHelpers(unittest.TestCase):

    def test_hash_prompt_is_sha256(self):
        prompt = "Test medical prompt"
        result = hash_prompt(prompt)
        expected = hashlib.sha256(prompt.encode()).hexdigest()
        self.assertEqual(result, expected)
        self.assertEqual(len(result), 64)

    def test_hash_prompt_deterministic(self):
        self.assertEqual(hash_prompt("same"), hash_prompt("same"))

    def test_hash_prompt_differs_for_different_inputs(self):
        self.assertNotEqual(hash_prompt("a"), hash_prompt("b"))

    def test_hash_generation_params_deterministic(self):
        h1 = hash_generation_params(temperature=0.3, max_tokens=500, model_id="m1")
        h2 = hash_generation_params(temperature=0.3, max_tokens=500, model_id="m1")
        self.assertEqual(h1, h2)

    def test_hash_generation_params_changes_on_diff_inputs(self):
        h1 = hash_generation_params(temperature=0.3, max_tokens=500, model_id="m1")
        h2 = hash_generation_params(temperature=0.5, max_tokens=500, model_id="m1")
        self.assertNotEqual(h1, h2)

    def test_hash_retrieval_params_deterministic(self):
        h1 = hash_retrieval_params(["id1", "id2"], [0.9, 0.8], top_k=3)
        h2 = hash_retrieval_params(["id1", "id2"], [0.9, 0.8], top_k=3)
        self.assertEqual(h1, h2)

    def test_hash_retrieval_params_order_sensitive(self):
        h1 = hash_retrieval_params(["id1", "id2"], [0.9, 0.8], top_k=3)
        h2 = hash_retrieval_params(["id2", "id1"], [0.8, 0.9], top_k=3)
        self.assertNotEqual(h1, h2)

    def test_hash_model_version_deterministic(self):
        h1 = hash_model_version("v1.0", "abc123")
        h2 = hash_model_version("v1.0", "abc123")
        self.assertEqual(h1, h2)

    def test_hash_model_version_empty_hash(self):
        result = hash_model_version("v1.0")
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)


# ---------------------------------------------------------------------------
# Bundle construction tests
# ---------------------------------------------------------------------------

class TestBundleConstruction(unittest.TestCase):

    def test_bundle_has_required_fields(self):
        bundle = _make_bundle()
        expected_keys = {
            "bundle_version", "timestamp", "hospital_id", "site_id",
            "device_id", "model_version_hash", "knowledge_base_hash",
            "explanation_hash", "retrieval_hash", "prompt_hash",
            "generation_params_hash", "bundle_hash",
        }
        self.assertTrue(expected_keys.issubset(set(bundle.keys())))

    def test_bundle_hash_is_sha256_length(self):
        bundle = _make_bundle()
        self.assertEqual(len(bundle["bundle_hash"]), 64)

    def test_bundle_hash_deterministic_same_inputs(self):
        b1 = _make_bundle()
        b2 = _make_bundle()
        self.assertEqual(b1["bundle_hash"], b2["bundle_hash"])

    def test_bundle_hash_changes_when_field_changes(self):
        b1 = _make_bundle(hospital_id="h1")
        b2 = _make_bundle(hospital_id="h2")
        self.assertNotEqual(b1["bundle_hash"], b2["bundle_hash"])

    def test_compute_bundle_hash_excludes_bundle_hash_field(self):
        bundle = _make_bundle()
        # Adding bundle_hash back should not change the recomputed value
        recomputed = compute_bundle_hash(bundle)
        self.assertEqual(recomputed, bundle["bundle_hash"])

    def test_bundle_version_stored(self):
        bundle = _make_bundle(bundle_version="2.0")
        self.assertEqual(bundle["bundle_version"], "2.0")

    def test_timestamp_default_is_recent(self):
        before = time.time()
        bundle = build_provenance_bundle(
            knowledge_base_hash=_DUMMY_HASH,
            explanation_hash=_DUMMY_HASH,
            retrieval_hash=_DUMMY_HASH,
            model_version_hash=_DUMMY_HASH,
            prompt_hash=_DUMMY_HASH,
            generation_params_hash=_DUMMY_HASH,
        )
        after = time.time()
        self.assertGreaterEqual(bundle["timestamp"], before)
        self.assertLessEqual(bundle["timestamp"], after)


# ---------------------------------------------------------------------------
# Signature verification tests
# ---------------------------------------------------------------------------

class TestSignatureVerification(unittest.TestCase):

    def setUp(self):
        """Generate a test key pair using eth_account."""
        try:
            from eth_account import Account
            # Deterministic test-only private key. Never use simple patterns
            # like this in production deployments.
            self.account = Account.from_key(
                "0x" + "1" * 64
            )
            self.private_key = "0x" + "1" * 64
            self.address = self.account.address
        except ImportError:
            self.skipTest("eth_account not available")

    def _sign(self, message: str) -> str:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        msg = encode_defunct(text=message)
        signed = Account.sign_message(msg, private_key=self.private_key)
        return signed.signature.hex()

    def test_valid_signature_returns_true(self):
        bundle = _make_bundle()
        sig = self._sign(bundle["bundle_hash"])
        self.assertTrue(verify_signature(bundle["bundle_hash"], sig, self.address))

    def test_wrong_signer_returns_false(self):
        bundle = _make_bundle()
        sig = self._sign(bundle["bundle_hash"])
        wrong_address = "0x" + "2" * 40
        self.assertFalse(verify_signature(bundle["bundle_hash"], sig, wrong_address))

    def test_tampered_hash_returns_false(self):
        bundle = _make_bundle()
        sig = self._sign(bundle["bundle_hash"])
        # Change one character in bundle_hash
        tampered = "b" * 64
        self.assertFalse(verify_signature(tampered, sig, self.address))

    def test_invalid_signature_returns_false(self):
        bundle = _make_bundle()
        result = verify_signature(bundle["bundle_hash"], "0x" + "ff" * 65, self.address)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Bundle integrity verification tests
# ---------------------------------------------------------------------------

class TestVerifyBundle(unittest.TestCase):

    def test_valid_bundle_hash(self):
        bundle = _make_bundle()
        result = verify_bundle(bundle)
        self.assertTrue(result["hash_valid"])

    def test_tampered_bundle_field_detected(self):
        bundle = _make_bundle()
        # Tamper with a field after building
        bundle["hospital_id"] = "tampered"
        result = verify_bundle(bundle)
        self.assertFalse(result["hash_valid"])

    def test_missing_bundle_hash_field(self):
        bundle = _make_bundle()
        del bundle["bundle_hash"]
        result = verify_bundle(bundle)
        self.assertFalse(result["hash_valid"])

    def test_signature_checked_when_provided(self):
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct
        except ImportError:
            self.skipTest("eth_account not available")

        private_key = "0x" + "3" * 64
        account = Account.from_key(private_key)
        bundle = _make_bundle()
        msg = encode_defunct(text=bundle["bundle_hash"])
        sig = Account.sign_message(msg, private_key=private_key).signature.hex()

        result = verify_bundle(bundle, signature=sig, signer_address=account.address)
        self.assertTrue(result["hash_valid"])
        self.assertTrue(result["signature_valid"])

    def test_signature_not_checked_when_not_provided(self):
        bundle = _make_bundle()
        result = verify_bundle(bundle)
        self.assertNotIn("signature_valid", result)


# ---------------------------------------------------------------------------
# ProvenanceIntegrator mock tests (no Ganache required)
# ---------------------------------------------------------------------------

class TestProvenanceIntegratorMock(unittest.TestCase):

    def setUp(self):
        try:
            from provenance_integrator import ProvenanceIntegrator
            self.ProvenanceIntegrator = ProvenanceIntegrator
        except ImportError:
            self.skipTest("provenance_integrator dependencies not available")

    def _get_integrator(self):
        return self.ProvenanceIntegrator(use_mock=True)

    def test_anchor_returns_tx_hash(self):
        bundle = _make_bundle()
        integrator = self._get_integrator()
        tx = integrator.anchor_provenance(
            bundle_hash=bundle["bundle_hash"],
            model_hash=bundle["model_version_hash"],
            kb_hash=bundle["knowledge_base_hash"],
            explanation_hash=bundle["explanation_hash"],
            signer_address="0x" + "a" * 40,
        )
        self.assertIsInstance(tx, str)
        self.assertTrue(tx.startswith("0x"))

    def test_is_anchored_after_anchor(self):
        bundle = _make_bundle()
        integrator = self._get_integrator()
        integrator.anchor_provenance(
            bundle_hash=bundle["bundle_hash"],
            model_hash=bundle["model_version_hash"],
            kb_hash=bundle["knowledge_base_hash"],
            explanation_hash=bundle["explanation_hash"],
            signer_address="0x" + "a" * 40,
        )
        self.assertTrue(integrator.is_anchored(bundle["bundle_hash"]))

    def test_is_anchored_false_for_unknown(self):
        integrator = self._get_integrator()
        self.assertFalse(integrator.is_anchored("f" * 64))

    def test_get_anchor_returns_details(self):
        bundle = _make_bundle()
        signer = "0x" + "b" * 40
        integrator = self._get_integrator()
        integrator.anchor_provenance(
            bundle_hash=bundle["bundle_hash"],
            model_hash=bundle["model_version_hash"],
            kb_hash=bundle["knowledge_base_hash"],
            explanation_hash=bundle["explanation_hash"],
            signer_address=signer,
        )
        anchor = integrator.get_anchor(bundle["bundle_hash"])
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["signer"], signer)

    def test_tx_hash_deterministic_for_same_bundle(self):
        bundle = _make_bundle()
        i1 = self._get_integrator()
        i2 = self._get_integrator()
        tx1 = i1.anchor_provenance(
            bundle_hash=bundle["bundle_hash"],
            model_hash=bundle["model_version_hash"],
            kb_hash=bundle["knowledge_base_hash"],
            explanation_hash=bundle["explanation_hash"],
            signer_address="0x" + "c" * 40,
        )
        tx2 = i2.anchor_provenance(
            bundle_hash=bundle["bundle_hash"],
            model_hash=bundle["model_version_hash"],
            kb_hash=bundle["knowledge_base_hash"],
            explanation_hash=bundle["explanation_hash"],
            signer_address="0x" + "c" * 40,
        )
        self.assertEqual(tx1, tx2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests():
    """Run all provenance tests."""
    print("Running Provenance Tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestHashHelpers))
    suite.addTests(loader.loadTestsFromTestCase(TestBundleConstruction))
    suite.addTests(loader.loadTestsFromTestCase(TestSignatureVerification))
    suite.addTests(loader.loadTestsFromTestCase(TestVerifyBundle))
    suite.addTests(loader.loadTestsFromTestCase(TestProvenanceIntegratorMock))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"Tests run:  {result.testsRun}")
    print(f"Failures:   {len(result.failures)}")
    print(f"Errors:     {len(result.errors)}")
    print(f"Skipped:    {len(result.skipped)}")
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
