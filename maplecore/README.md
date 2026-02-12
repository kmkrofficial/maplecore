# MAPLE Core SDK

The core Python SDK powering the efficient retrieval engine.

## Modules
- **`indexer.py` (`MapleIndexer`)**: Handles document processing, chunking, and embedding generation using BGE-Small.
- **`scanner.py` (`MapleScanner`)**: Implements retrieval algorithms (Linear, Hierarchical, Adaptive) to find relevant context blocks.
- **`core.py` / `model.py` (`MapleNet`)**: Defines the lightweight MLP architecture that predicts block relevance.
- **`client.py` (`Maple`)**: High-level client API for easy integration.
- **`trainer.py` (`MapleTrainer`)**: Training logic for optimizing the predictive model.
- **`utils.py`**: Common utilities for device management and logging.
