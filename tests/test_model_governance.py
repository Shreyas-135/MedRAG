"""
Tests for Model Governance Integrator

Tests:
- ModelGovernanceIntegrator mock mode (no Ganache required)
- compute_model_version_hash determinism
- register_model and approve_model flows
- 3-of-4 approval threshold (PENDING → APPROVED)
- double-approval rejection
- get_approval_status responses
"""

import sys
import os
import hashlib
import json
import unittest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestModelVersionHash(unittest.TestCase):
    """Verify compute_model_version_hash is deterministic."""

    def setUp(self):
        try:
            from model_governance_integrator import ModelGovernanceIntegrator
            self.ModelGovernanceIntegrator = ModelGovernanceIntegrator
        except ImportError:
            self.skipTest("model_governance_integrator not available")

    def test_hash_deterministic_same_inputs(self):
        h1 = self.ModelGovernanceIntegrator.compute_model_version_hash("v1.0", "abc123")
        h2 = self.ModelGovernanceIntegrator.compute_model_version_hash("v1.0", "abc123")
        self.assertEqual(h1, h2)

    def test_hash_length(self):
        h = self.ModelGovernanceIntegrator.compute_model_version_hash("v1.0")
        self.assertEqual(len(h), 64)

    def test_hash_differs_for_different_version_ids(self):
        h1 = self.ModelGovernanceIntegrator.compute_model_version_hash("v1.0")
        h2 = self.ModelGovernanceIntegrator.compute_model_version_hash("v2.0")
        self.assertNotEqual(h1, h2)

    def test_hash_differs_for_different_model_hashes(self):
        h1 = self.ModelGovernanceIntegrator.compute_model_version_hash("v1.0", "aaa")
        h2 = self.ModelGovernanceIntegrator.compute_model_version_hash("v1.0", "bbb")
        self.assertNotEqual(h1, h2)

    def test_hash_consistent_with_direct_sha256(self):
        payload = {"version_id": "v1.0", "model_hash": ""}
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(content.encode()).hexdigest()
        result = self.ModelGovernanceIntegrator.compute_model_version_hash("v1.0")
        self.assertEqual(result, expected)


class TestGovernanceIntegratorMock(unittest.TestCase):
    """Test MockGovernanceIntegrator in-process (no Ganache required)."""

    TEST_MODEL_HASH = "a" * 64

    def setUp(self):
        try:
            from model_governance_integrator import ModelGovernanceIntegrator
            self.integrator = ModelGovernanceIntegrator(use_mock=True, required_approvals=3)
        except ImportError:
            self.skipTest("model_governance_integrator not available")

    def test_initial_status_unknown(self):
        status = self.integrator.get_approval_status(self.TEST_MODEL_HASH)
        self.assertEqual(status["status"], "UNKNOWN")
        self.assertFalse(status["is_approved"])

    def test_register_model_returns_tx(self):
        tx = self.integrator.register_model(self.TEST_MODEL_HASH)
        self.assertIsInstance(tx, str)
        self.assertTrue(tx.startswith("0x"))

    def test_status_pending_after_register(self):
        self.integrator.register_model(self.TEST_MODEL_HASH)
        status = self.integrator.get_approval_status(self.TEST_MODEL_HASH)
        self.assertEqual(status["status"], "PENDING")
        self.assertFalse(status["is_approved"])
        self.assertEqual(status["approval_count"], 0)

    def test_double_register_raises(self):
        self.integrator.register_model(self.TEST_MODEL_HASH)
        with self.assertRaises(ValueError):
            self.integrator.register_model(self.TEST_MODEL_HASH)

    def test_single_approval_still_pending(self):
        self.integrator.register_model(self.TEST_MODEL_HASH)
        self.integrator.approve_model(self.TEST_MODEL_HASH, hospital_key="key_hospital_1")
        status = self.integrator.get_approval_status(self.TEST_MODEL_HASH)
        self.assertEqual(status["status"], "PENDING")
        self.assertEqual(status["approval_count"], 1)

    def test_three_approvals_become_approved(self):
        self.integrator.register_model(self.TEST_MODEL_HASH)
        for i in range(3):
            self.integrator.approve_model(self.TEST_MODEL_HASH, hospital_key=f"key_hospital_{i}")
        self.assertTrue(self.integrator.is_approved(self.TEST_MODEL_HASH))
        status = self.integrator.get_approval_status(self.TEST_MODEL_HASH)
        self.assertEqual(status["status"], "APPROVED")

    def test_four_approvals_one_redundant(self):
        """4th approval attempt on an already-APPROVED model raises ValueError."""
        self.integrator.register_model(self.TEST_MODEL_HASH)
        for i in range(3):
            self.integrator.approve_model(self.TEST_MODEL_HASH, hospital_key=f"key_hospital_{i}")
        self.assertTrue(self.integrator.is_approved(self.TEST_MODEL_HASH))
        # Contract rejects approvals once APPROVED (mirrors Solidity behaviour)
        with self.assertRaises(ValueError):
            self.integrator.approve_model(self.TEST_MODEL_HASH, hospital_key="key_hospital_3")

    def test_double_approval_same_hospital_rejected(self):
        self.integrator.register_model(self.TEST_MODEL_HASH)
        self.integrator.approve_model(self.TEST_MODEL_HASH, hospital_key="key_hospital_1")
        with self.assertRaises(ValueError):
            self.integrator.approve_model(self.TEST_MODEL_HASH, hospital_key="key_hospital_1")

    def test_approve_unregistered_model_raises(self):
        with self.assertRaises(ValueError):
            self.integrator.approve_model("b" * 64, hospital_key="key_hospital_1")

    def test_is_approved_false_for_unknown(self):
        self.assertFalse(self.integrator.is_approved("c" * 64))

    def test_approval_tx_deterministic(self):
        """Same hospital key → same fake TX hash."""
        integrator2 = type(self.integrator)(use_mock=True, required_approvals=3)
        self.integrator.register_model(self.TEST_MODEL_HASH)
        integrator2.register_model(self.TEST_MODEL_HASH)
        tx1 = self.integrator.approve_model(self.TEST_MODEL_HASH, hospital_key="key_h1")
        tx2 = integrator2.approve_model(self.TEST_MODEL_HASH, hospital_key="key_h1")
        self.assertEqual(tx1, tx2)

    def test_required_approvals_configurable(self):
        from model_governance_integrator import ModelGovernanceIntegrator
        integrator = ModelGovernanceIntegrator(use_mock=True, required_approvals=2)
        mh = "d" * 64
        integrator.register_model(mh)
        integrator.approve_model(mh, hospital_key="key_hospital_0")
        self.assertFalse(integrator.is_approved(mh))
        integrator.approve_model(mh, hospital_key="key_hospital_1")
        self.assertTrue(integrator.is_approved(mh))

    def test_required_approvals_in_status(self):
        status = self.integrator.get_approval_status("e" * 64)
        self.assertEqual(status["required_approvals"], 3)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests():
    """Run all governance tests."""
    print("Running Model Governance Tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestModelVersionHash))
    suite.addTests(loader.loadTestsFromTestCase(TestGovernanceIntegratorMock))

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
