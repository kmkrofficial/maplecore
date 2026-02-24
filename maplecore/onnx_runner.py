"""
MAPLE ONNX Production Runner
============================
Executes ultra-fast inference sequentially without the PyTorch overhead.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

class MapleONNXRunner:
    """
    ONNX Runtime executor for MAPLE.
    Uses ONNX inference sessions for faster CPU-bound production environments.
    """
    
    def __init__(self, model_path: Union[str, Path]) -> None:
        """
        Load the ONNX MAPLE Session.
        
        Args:
            model_path: Path to exported .onnx file
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("ONNX Runtime not installed. Install with `pip install maplecore[production]`")
            raise
            
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX Model file missing: {self.model_path}")
            
        logger.info(f"Initializing ONNX InferenceSession against {self.model_path}")
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference and return relevance scores.
        Applies mathematical sigmoid over the ONNX-emitted logits.
        
        Args:
            x: Numpy matrix array of [batch_size, input_dim]
            
        Returns:
            Sigmoid output scoring array [batch_size]
        """
        # ONNX execution
        logits = self.session.run(None, {self.input_name: x})[0]
        
        # Apply Sigmoid locally: 1 / (1 + exp(-x))
        scores = 1 / (1 + np.exp(-logits))
        return scores.squeeze()
