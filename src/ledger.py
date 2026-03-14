"""
Comprehensive Ledger System for MedRAG
Provides immutable logging of training rounds and access events.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd


class LedgerEntry:
    """Base class for ledger entries."""
    
    def __init__(self, entry_type: str, timestamp: str = None):
        self.entry_type = entry_type
        self.timestamp = timestamp or datetime.now().isoformat()
        self.entry_hash = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        raise NotImplementedError
    
    def compute_hash(self, previous_hash: str = "0") -> str:
        """Compute hash of entry with previous hash (blockchain-style)."""
        content = json.dumps(self.to_dict(), sort_keys=True) + previous_hash
        return hashlib.sha256(content.encode()).hexdigest()


class TrainingRoundEntry(LedgerEntry):
    """Entry for a training round."""
    
    def __init__(self, round_num: int, node_metrics: Dict[str, Dict], 
                 model_hash: str, blockchain_tx: Optional[str] = None,
                 rag_retrieval_count: int = 0, privacy_budget: float = 0.0,
                 timestamp: str = None):
        super().__init__("training_round", timestamp)
        self.round_num = round_num
        self.node_metrics = node_metrics  # {node_id: {loss: x, accuracy: y}}
        self.model_hash = model_hash
        self.blockchain_tx = blockchain_tx
        self.rag_retrieval_count = rag_retrieval_count
        self.privacy_budget = privacy_budget
    
    def to_dict(self) -> Dict:
        return {
            'entry_type': self.entry_type,
            'timestamp': self.timestamp,
            'round_num': self.round_num,
            'node_metrics': self.node_metrics,
            'model_hash': self.model_hash,
            'blockchain_tx': self.blockchain_tx,
            'rag_retrieval_count': self.rag_retrieval_count,
            'privacy_budget': self.privacy_budget,
            'entry_hash': self.entry_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingRoundEntry':
        entry = cls(
            round_num=data['round_num'],
            node_metrics=data['node_metrics'],
            model_hash=data['model_hash'],
            blockchain_tx=data.get('blockchain_tx'),
            rag_retrieval_count=data.get('rag_retrieval_count', 0),
            privacy_budget=data.get('privacy_budget', 0.0),
            timestamp=data['timestamp']
        )
        entry.entry_hash = data.get('entry_hash')
        return entry


class AccessLogEntry(LedgerEntry):
    """Entry for access/action logs."""
    
    def __init__(self, user_id: str, action: str, resource: str, 
                 status: str, details: Optional[Dict] = None, 
                 timestamp: str = None):
        super().__init__("access_log", timestamp)
        self.user_id = user_id
        self.action = action  # train, predict, access_model, view_data
        self.resource = resource
        self.status = status  # success, failure
        self.details = details or {}
    
    def to_dict(self) -> Dict:
        return {
            'entry_type': self.entry_type,
            'timestamp': self.timestamp,
            'user_id': self.user_id,
            'action': self.action,
            'resource': self.resource,
            'status': self.status,
            'details': self.details,
            'entry_hash': self.entry_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AccessLogEntry':
        entry = cls(
            user_id=data['user_id'],
            action=data['action'],
            resource=data['resource'],
            status=data['status'],
            details=data.get('details', {}),
            timestamp=data['timestamp']
        )
        entry.entry_hash = data.get('entry_hash')
        return entry


class Ledger:
    """
    Immutable ledger for logging training and access events.
    
    Features:
    - Append-only logging
    - Hash chaining for integrity verification
    - Training round logging with metrics
    - Access event logging
    - Export to JSON/CSV
    """
    
    def __init__(self, ledger_dir: str = None):
        """
        Initialize ledger.
        
        Args:
            ledger_dir: Directory to store ledger files.
                       Defaults to 'ledger' relative to project root.
        """
        if ledger_dir is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            ledger_dir = os.path.join(repo_root, 'ledger')
        
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_log_file = self.ledger_dir / 'training_log.jsonl'
        self.access_log_file = self.ledger_dir / 'access_log.jsonl'
        
        # Initialize hash chains
        self.training_last_hash = "0"
        self.access_last_hash = "0"
        
        # Load existing hashes
        self._load_last_hashes()
    
    def _load_last_hashes(self):
        """Load the last hash from each log file."""
        # Training log
        if self.training_log_file.exists():
            try:
                with open(self.training_log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_entry = json.loads(lines[-1])
                        self.training_last_hash = last_entry.get('entry_hash', '0')
            except Exception as e:
                print(f"Warning: Could not load training log hash: {e}")
        
        # Access log
        if self.access_log_file.exists():
            try:
                with open(self.access_log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_entry = json.loads(lines[-1])
                        self.access_last_hash = last_entry.get('entry_hash', '0')
            except Exception as e:
                print(f"Warning: Could not load access log hash: {e}")
    
    def _append_to_log(self, entry: LedgerEntry, log_file: Path, 
                       last_hash: str) -> str:
        """
        Append entry to log file with hash chaining.
        
        Args:
            entry: LedgerEntry to append
            log_file: Path to log file
            last_hash: Previous entry hash
            
        Returns:
            New entry hash
        """
        # Compute hash
        entry.entry_hash = entry.compute_hash(last_hash)
        
        # Append to file (JSONL format)
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
        
        return entry.entry_hash
    
    def log_training_round(self, round_num: int, node_metrics: Dict[str, Dict],
                          model_hash: str, blockchain_tx: Optional[str] = None,
                          rag_retrieval_count: int = 0, privacy_budget: float = 0.0):
        """
        Log a training round.
        
        Args:
            round_num: Round number
            node_metrics: Metrics per node/client
                         e.g., {'client0': {'loss': 0.5, 'accuracy': 0.8}, ...}
            model_hash: SHA-256 hash of model weights
            blockchain_tx: Optional blockchain transaction hash
            rag_retrieval_count: Number of RAG retrievals
            privacy_budget: Privacy budget used (epsilon)
        """
        entry = TrainingRoundEntry(
            round_num=round_num,
            node_metrics=node_metrics,
            model_hash=model_hash,
            blockchain_tx=blockchain_tx,
            rag_retrieval_count=rag_retrieval_count,
            privacy_budget=privacy_budget
        )
        
        self.training_last_hash = self._append_to_log(
            entry, self.training_log_file, self.training_last_hash
        )
    
    def log_access(self, user_id: str, action: str, resource: str, 
                   status: str, details: Optional[Dict] = None):
        """
        Log an access/action event.
        
        Args:
            user_id: User or client identifier
            action: Action type (train, predict, access_model, view_data)
            resource: Resource accessed
            status: success or failure
            details: Optional additional details
        """
        entry = AccessLogEntry(
            user_id=user_id,
            action=action,
            resource=resource,
            status=status,
            details=details
        )
        
        self.access_last_hash = self._append_to_log(
            entry, self.access_log_file, self.access_last_hash
        )
    
    def get_training_history(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get training history as DataFrame.
        
        Args:
            limit: Optional limit on number of entries (most recent)
            
        Returns:
            DataFrame with training history
        """
        if not self.training_log_file.exists():
            return pd.DataFrame()
        
        entries = []
        with open(self.training_log_file, 'r') as f:
            for line in f:
                entry_dict = json.loads(line)
                entries.append(entry_dict)
        
        if limit:
            entries = entries[-limit:]
        
        if not entries:
            return pd.DataFrame()
        
        # Flatten for DataFrame
        rows = []
        for entry in entries:
            row = {
                'timestamp': entry['timestamp'],
                'round_num': entry['round_num'],
                'model_hash': entry['model_hash'][:16] + '...',  # Truncate for display
                'blockchain_tx': entry.get('blockchain_tx', 'N/A'),
                'rag_retrieval_count': entry.get('rag_retrieval_count', 0),
                'privacy_budget': entry.get('privacy_budget', 0.0),
            }
            
            # Add node metrics
            for node_id, metrics in entry.get('node_metrics', {}).items():
                for metric_name, metric_value in metrics.items():
                    row[f"{node_id}_{metric_name}"] = metric_value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_access_logs(self, user_id: Optional[str] = None, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       action: Optional[str] = None) -> pd.DataFrame:
        """
        Get access logs with optional filtering.
        
        Args:
            user_id: Filter by user ID
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            action: Filter by action type
            
        Returns:
            DataFrame with access logs
        """
        if not self.access_log_file.exists():
            return pd.DataFrame()
        
        entries = []
        with open(self.access_log_file, 'r') as f:
            for line in f:
                entry_dict = json.loads(line)
                
                # Apply filters
                if user_id and entry_dict.get('user_id') != user_id:
                    continue
                if action and entry_dict.get('action') != action:
                    continue
                if start_date and entry_dict.get('timestamp') < start_date:
                    continue
                if end_date and entry_dict.get('timestamp') > end_date:
                    continue
                
                entries.append(entry_dict)
        
        if not entries:
            return pd.DataFrame()
        
        return pd.DataFrame(entries)
    
    def export_ledger(self, output_path: str, format: str = 'json', 
                     ledger_type: str = 'training') -> str:
        """
        Export ledger to file.
        
        Args:
            output_path: Output file path
            format: 'json' or 'csv'
            ledger_type: 'training' or 'access'
            
        Returns:
            Path to exported file
        """
        if ledger_type == 'training':
            df = self.get_training_history()
        else:
            df = self.get_access_logs()
        
        if df.empty:
            print(f"No {ledger_type} entries to export")
            return None
        
        output_path = Path(output_path)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(output_path)
    
    def verify_integrity(self, ledger_type: str = 'training') -> bool:
        """
        Verify integrity of ledger by checking hash chain.
        
        Args:
            ledger_type: 'training' or 'access'
            
        Returns:
            True if integrity verified, False otherwise
        """
        log_file = (self.training_log_file if ledger_type == 'training' 
                   else self.access_log_file)
        
        if not log_file.exists():
            return True  # Empty ledger is valid
        
        previous_hash = "0"
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    entry_dict = json.loads(line)
                    stored_hash = entry_dict.get('entry_hash')
                    
                    # Recreate entry to compute hash
                    if ledger_type == 'training':
                        entry = TrainingRoundEntry.from_dict(entry_dict)
                    else:
                        entry = AccessLogEntry.from_dict(entry_dict)
                    
                    computed_hash = entry.compute_hash(previous_hash)
                    
                    if computed_hash != stored_hash:
                        return False
                    
                    previous_hash = stored_hash
            
            return True
        except Exception as e:
            print(f"Error verifying integrity: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the ledger."""
        training_count = 0
        access_count = 0
        
        if self.training_log_file.exists():
            with open(self.training_log_file, 'r') as f:
                training_count = sum(1 for _ in f)
        
        if self.access_log_file.exists():
            with open(self.access_log_file, 'r') as f:
                access_count = sum(1 for _ in f)
        
        return {
            'training_entries': training_count,
            'access_entries': access_count,
            'training_integrity': self.verify_integrity('training'),
            'access_integrity': self.verify_integrity('access')
        }


if __name__ == "__main__":
    # Example usage
    print("Ledger System Demo")
    print("=" * 50)
    
    # Initialize ledger
    ledger = Ledger()
    print(f"✓ Ledger initialized at: {ledger.ledger_dir}")
    
    # Get summary
    summary = ledger.get_summary()
    print(f"\nLedger Summary:")
    print(f"  Training entries: {summary['training_entries']}")
    print(f"  Access entries: {summary['access_entries']}")
    print(f"  Training integrity: {'✓ Valid' if summary['training_integrity'] else '✗ Invalid'}")
    print(f"  Access integrity: {'✓ Valid' if summary['access_integrity'] else '✗ Invalid'}")
