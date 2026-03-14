"""
RAG Configuration Module
Centralized configuration for LangChain RAG, ChromaDB, Gemini, and Flower.
"""
import os
from typing import Optional


class RAGConfig:
    """Configuration for RAG components"""
    
    # ChromaDB Settings
    CHROMADB_PERSIST_DIR = os.getenv('CHROMADB_PERSIST_DIR', './chromadb_data')
    CHROMADB_COLLECTION = os.getenv('CHROMADB_COLLECTION', 'medical_xray_findings')
    CHROMADB_SIMILARITY_METRIC = 'cosine'
    CHROMADB_TOP_K = 5
    
    # Gemini Settings
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
    GEMINI_TEMPERATURE = float(os.getenv('GEMINI_TEMPERATURE', '0.3'))
    GEMINI_MAX_TOKENS = int(os.getenv('GEMINI_MAX_TOKENS', '500'))
    GEMINI_TOP_P = float(os.getenv('GEMINI_TOP_P', '0.95'))
    GEMINI_TOP_K = int(os.getenv('GEMINI_TOP_K', '40'))
    
    # Flower Federated Learning Settings
    FLOWER_SERVER_ADDRESS = os.getenv('FLOWER_SERVER_ADDRESS', '[::]:8080')
    FLOWER_NUM_ROUNDS = int(os.getenv('FLOWER_NUM_ROUNDS', '10'))
    FLOWER_MIN_CLIENTS = int(os.getenv('FLOWER_MIN_CLIENTS', '4'))
    FLOWER_MIN_AVAILABLE_CLIENTS = int(os.getenv('FLOWER_MIN_AVAILABLE_CLIENTS', '4'))
    FLOWER_FIT_CONFIG = {
        'batch_size': 32,
        'local_epochs': 5,
    }
    
    # RAG Pipeline Settings
    RAG_RETRIEVAL_TOP_K = 3
    RAG_CITATION_FORMAT = 'clinical'
    RAG_EXPLANATION_MAX_LENGTH = 500
    RAG_USE_RERANKING = False
    
    # Medical Knowledge Base Settings
    KB_EMBEDDING_DIM = 64
    KB_INDEX_TYPE = 'cosine'
    KB_BATCH_SIZE = 100
    
    @classmethod
    def validate(cls) -> tuple[bool, Optional[str]]:
        """
        Validate configuration settings.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == 'your-gemini-api-key-here':
            return False, "GEMINI_API_KEY not set. Please set it in environment or .env file."
        
        if cls.GEMINI_TEMPERATURE < 0 or cls.GEMINI_TEMPERATURE > 1:
            return False, "GEMINI_TEMPERATURE must be between 0 and 1."
        
        if cls.FLOWER_NUM_ROUNDS < 1:
            return False, "FLOWER_NUM_ROUNDS must be at least 1."
        
        if cls.FLOWER_MIN_CLIENTS < 1:
            return False, "FLOWER_MIN_CLIENTS must be at least 1."
        
        return True, None
    
    @classmethod
    def get_summary(cls) -> dict:
        """Get configuration summary."""
        return {
            'chromadb': {
                'persist_dir': cls.CHROMADB_PERSIST_DIR,
                'collection': cls.CHROMADB_COLLECTION,
                'top_k': cls.CHROMADB_TOP_K,
            },
            'gemini': {
                'model': cls.GEMINI_MODEL,
                'temperature': cls.GEMINI_TEMPERATURE,
                'max_tokens': cls.GEMINI_MAX_TOKENS,
                'api_key_set': bool(cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != 'your-gemini-api-key-here'),
            },
            'flower': {
                'server_address': cls.FLOWER_SERVER_ADDRESS,
                'num_rounds': cls.FLOWER_NUM_ROUNDS,
                'min_clients': cls.FLOWER_MIN_CLIENTS,
            },
            'rag': {
                'retrieval_top_k': cls.RAG_RETRIEVAL_TOP_K,
                'citation_format': cls.RAG_CITATION_FORMAT,
            }
        }


class FlowerConfig:
    """Flower-specific configuration"""
    
    # Client Configuration
    CLIENT_TIMEOUT_SECONDS = 300
    CLIENT_MAX_RETRIES = 3
    
    # Server Configuration
    SERVER_ROUND_TIMEOUT = 600
    SERVER_STRATEGY = 'FedAvg'
    
    # Differential Privacy for VFL
    DP_ENABLED = True
    DP_NOISE_MULTIPLIER = 0.1
    DP_CLIPPING_NORM = 1.0
    
    # VFL-specific
    VFL_NUM_CLIENTS = 4
    VFL_EMBEDDING_DIM = 64
    VFL_AGGREGATION_METHOD = 'sum'


if __name__ == '__main__':
    # Print configuration summary
    print("RAG Configuration Summary")
    print("=" * 60)
    
    is_valid, error = RAGConfig.validate()
    if is_valid:
        print("✓ Configuration is valid")
    else:
        print(f"✗ Configuration error: {error}")
    
    print("\nSettings:")
    summary = RAGConfig.get_summary()
    for category, settings in summary.items():
        print(f"\n{category.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
