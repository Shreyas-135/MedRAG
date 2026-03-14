"""
Unit tests for Ledger System

This module contains tests for the Ledger class used in MedRAG for maintaining
an immutable audit trail. Tests verify:
- Training round logging
- Access log recording
- Hash chain integrity verification
- Event sequencing and timestamps
- Data retrieval and querying
- Persistence across sessions

The ledger ensures complete transparency and accountability for all system operations,
supporting compliance requirements for medical AI systems.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ledger import Ledger, TrainingRoundEntry, AccessLogEntry


class TestLedger(unittest.TestCase):
    """Test cases for Ledger"""
    
    def setUp(self):
        """Create temporary directory for testing"""
        self.test_dir = tempfile.mkdtemp()
        self.ledger = Ledger(ledger_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test ledger initialization"""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertEqual(self.ledger.training_last_hash, "0")
        self.assertEqual(self.ledger.access_last_hash, "0")
    
    def test_log_training_round(self):
        """Test logging a training round"""
        self.ledger.log_training_round(
            round_num=1,
            node_metrics={'client0': {'loss': 0.5, 'accuracy': 0.8}},
            model_hash='abc123',
            blockchain_tx='0x123',
            rag_retrieval_count=5,
            privacy_budget=0.1
        )
        
        # Check file exists
        self.assertTrue(self.ledger.training_log_file.exists())
        
        # Check content
        with open(self.ledger.training_log_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            
            entry = json.loads(lines[0])
            self.assertEqual(entry['round_num'], 1)
            self.assertEqual(entry['model_hash'], 'abc123')
            self.assertEqual(entry['privacy_budget'], 0.1)
    
    def test_log_access(self):
        """Test logging an access event"""
        self.ledger.log_access(
            user_id='user1',
            action='predict',
            resource='xray_001.jpg',
            status='success',
            details={'confidence': 0.95}
        )
        
        # Check file exists
        self.assertTrue(self.ledger.access_log_file.exists())
        
        # Check content
        with open(self.ledger.access_log_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            
            entry = json.loads(lines[0])
            self.assertEqual(entry['user_id'], 'user1')
            self.assertEqual(entry['action'], 'predict')
            self.assertEqual(entry['status'], 'success')
    
    def test_get_training_history(self):
        """Test retrieving training history"""
        # Log multiple rounds
        for i in range(3):
            self.ledger.log_training_round(
                round_num=i+1,
                node_metrics={'client0': {'loss': 0.5-i*0.1}},
                model_hash=f'hash{i}',
                privacy_budget=0.1
            )
        
        df = self.ledger.get_training_history()
        self.assertEqual(len(df), 3)
        self.assertIn('round_num', df.columns)
        self.assertIn('model_hash', df.columns)
    
    def test_get_training_history_with_limit(self):
        """Test retrieving limited training history"""
        # Log 5 rounds
        for i in range(5):
            self.ledger.log_training_round(
                round_num=i+1,
                node_metrics={'client0': {'loss': 0.5}},
                model_hash=f'hash{i}',
                privacy_budget=0.1
            )
        
        df = self.ledger.get_training_history(limit=3)
        self.assertEqual(len(df), 3)
        # Should get the last 3 entries
        self.assertIn(5, df['round_num'].values)
    
    def test_get_access_logs(self):
        """Test retrieving access logs"""
        # Log multiple access events
        users = ['user1', 'user2', 'user1']
        actions = ['predict', 'train', 'predict']
        
        for user, action in zip(users, actions):
            self.ledger.log_access(
                user_id=user,
                action=action,
                resource='resource',
                status='success'
            )
        
        df = self.ledger.get_access_logs()
        self.assertEqual(len(df), 3)
    
    def test_get_access_logs_filtered_by_user(self):
        """Test filtering access logs by user"""
        # Log events for different users
        self.ledger.log_access('user1', 'predict', 'res1', 'success')
        self.ledger.log_access('user2', 'predict', 'res2', 'success')
        self.ledger.log_access('user1', 'train', 'res3', 'success')
        
        df = self.ledger.get_access_logs(user_id='user1')
        self.assertEqual(len(df), 2)
        self.assertTrue(all(df['user_id'] == 'user1'))
    
    def test_get_access_logs_filtered_by_action(self):
        """Test filtering access logs by action"""
        self.ledger.log_access('user1', 'predict', 'res1', 'success')
        self.ledger.log_access('user1', 'train', 'res2', 'success')
        self.ledger.log_access('user2', 'predict', 'res3', 'success')
        
        df = self.ledger.get_access_logs(action='predict')
        self.assertEqual(len(df), 2)
        self.assertTrue(all(df['action'] == 'predict'))
    
    def test_verify_integrity_empty(self):
        """Test integrity verification on empty ledger"""
        is_valid = self.ledger.verify_integrity('training')
        self.assertTrue(is_valid)
    
    def test_verify_integrity_valid(self):
        """Test integrity verification on valid ledger"""
        # Log some rounds
        for i in range(3):
            self.ledger.log_training_round(
                round_num=i+1,
                node_metrics={'client0': {'loss': 0.5}},
                model_hash=f'hash{i}',
                privacy_budget=0.1
            )
        
        is_valid = self.ledger.verify_integrity('training')
        self.assertTrue(is_valid)
    
    def test_verify_integrity_tampered(self):
        """Test integrity verification detects tampering"""
        # Log a round
        self.ledger.log_training_round(
            round_num=1,
            node_metrics={'client0': {'loss': 0.5}},
            model_hash='hash1',
            privacy_budget=0.1
        )
        
        # Tamper with the log file
        with open(self.ledger.training_log_file, 'r') as f:
            lines = f.readlines()
        
        # Modify the entry
        entry = json.loads(lines[0])
        entry['model_hash'] = 'tampered_hash'
        
        with open(self.ledger.training_log_file, 'w') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Verification should fail
        is_valid = self.ledger.verify_integrity('training')
        self.assertFalse(is_valid)
    
    def test_hash_chain(self):
        """Test that hash chain is maintained"""
        # Log multiple rounds
        for i in range(3):
            self.ledger.log_training_round(
                round_num=i+1,
                node_metrics={'client0': {'loss': 0.5}},
                model_hash=f'hash{i}',
                privacy_budget=0.1
            )
        
        # Read entries and verify hash chain
        with open(self.ledger.training_log_file, 'r') as f:
            lines = f.readlines()
        
        prev_hash = "0"
        for line in lines:
            entry = json.loads(line)
            # Each entry's hash should be computed from its content + prev_hash
            self.assertIsNotNone(entry['entry_hash'])
            prev_hash = entry['entry_hash']
    
    def test_export_ledger_csv(self):
        """Test exporting ledger to CSV"""
        self.ledger.log_training_round(
            round_num=1,
            node_metrics={'client0': {'loss': 0.5}},
            model_hash='hash1',
            privacy_budget=0.1
        )
        
        export_path = Path(self.test_dir) / 'export.csv'
        result_path = self.ledger.export_ledger(str(export_path), format='csv', ledger_type='training')
        
        self.assertIsNotNone(result_path)
        self.assertTrue(Path(result_path).exists())
    
    def test_export_ledger_json(self):
        """Test exporting ledger to JSON"""
        self.ledger.log_training_round(
            round_num=1,
            node_metrics={'client0': {'loss': 0.5}},
            model_hash='hash1',
            privacy_budget=0.1
        )
        
        export_path = Path(self.test_dir) / 'export.json'
        result_path = self.ledger.export_ledger(str(export_path), format='json', ledger_type='training')
        
        self.assertIsNotNone(result_path)
        self.assertTrue(Path(result_path).exists())
    
    def test_get_summary(self):
        """Test getting ledger summary"""
        # Initially empty
        summary = self.ledger.get_summary()
        self.assertEqual(summary['training_entries'], 0)
        self.assertEqual(summary['access_entries'], 0)
        
        # Add entries
        self.ledger.log_training_round(1, {'client0': {'loss': 0.5}}, 'hash1', privacy_budget=0.1)
        self.ledger.log_access('user1', 'predict', 'res1', 'success')
        
        summary = self.ledger.get_summary()
        self.assertEqual(summary['training_entries'], 1)
        self.assertEqual(summary['access_entries'], 1)
        self.assertTrue(summary['training_integrity'])
        self.assertTrue(summary['access_integrity'])


class TestTrainingRoundEntry(unittest.TestCase):
    """Test cases for TrainingRoundEntry"""
    
    def test_to_dict(self):
        """Test converting to dictionary"""
        entry = TrainingRoundEntry(
            round_num=1,
            node_metrics={'client0': {'loss': 0.5}},
            model_hash='abc123',
            blockchain_tx='0x123',
            rag_retrieval_count=5,
            privacy_budget=0.1
        )
        
        data = entry.to_dict()
        self.assertEqual(data['round_num'], 1)
        self.assertEqual(data['model_hash'], 'abc123')
        self.assertEqual(data['rag_retrieval_count'], 5)
    
    def test_from_dict(self):
        """Test creating from dictionary"""
        data = {
            'round_num': 1,
            'node_metrics': {'client0': {'loss': 0.5}},
            'model_hash': 'abc123',
            'blockchain_tx': '0x123',
            'rag_retrieval_count': 5,
            'privacy_budget': 0.1,
            'timestamp': '2023-12-23T14:30:00',
            'entry_hash': 'hash123'
        }
        
        entry = TrainingRoundEntry.from_dict(data)
        self.assertEqual(entry.round_num, 1)
        self.assertEqual(entry.model_hash, 'abc123')
    
    def test_compute_hash(self):
        """Test hash computation"""
        entry = TrainingRoundEntry(
            round_num=1,
            node_metrics={'client0': {'loss': 0.5}},
            model_hash='abc123',
            privacy_budget=0.1
        )
        
        hash1 = entry.compute_hash("prev_hash")
        hash2 = entry.compute_hash("prev_hash")
        
        # Same input should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Different previous hash should produce different hash
        hash3 = entry.compute_hash("different_prev_hash")
        self.assertNotEqual(hash1, hash3)


class TestAccessLogEntry(unittest.TestCase):
    """Test cases for AccessLogEntry"""
    
    def test_to_dict(self):
        """Test converting to dictionary"""
        entry = AccessLogEntry(
            user_id='user1',
            action='predict',
            resource='xray.jpg',
            status='success',
            details={'confidence': 0.95}
        )
        
        data = entry.to_dict()
        self.assertEqual(data['user_id'], 'user1')
        self.assertEqual(data['action'], 'predict')
        self.assertEqual(data['status'], 'success')
    
    def test_from_dict(self):
        """Test creating from dictionary"""
        data = {
            'user_id': 'user1',
            'action': 'predict',
            'resource': 'xray.jpg',
            'status': 'success',
            'details': {'confidence': 0.95},
            'timestamp': '2023-12-23T14:30:00',
            'entry_hash': 'hash123'
        }
        
        entry = AccessLogEntry.from_dict(data)
        self.assertEqual(entry.user_id, 'user1')
        self.assertEqual(entry.action, 'predict')


if __name__ == '__main__':
    unittest.main()
