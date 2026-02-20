#!/usr/bin/env python3
"""
Verify ONNX Export
==================
Tests if MAPLE can be effectively exported to an ONNX runtime
and whether inference matches standard PyTorch logit distributions.
"""

import os
import torch
import numpy as np
from pathlib import Path

from maplecore.core import MapleNet
from maplecore.onnx_runner import MapleONNXRunner
from maplecore.search import MapleScanner

def cleanup(path):
    if os.path.exists(path):
        os.remove(path)

def run_test():
    print("Initializing dummy PyTorch MapleNet...")
    # Initialize a dummy MAPLE net with 768 dims
    pt_model = MapleNet(input_dim=768, hidden_dim=128)
    pt_model.eval()
    
    # Export it
    onnx_path = "models/maple_test.onnx"
    print(f"Exporting to {onnx_path}...")
    pt_model.export_to_onnx(onnx_path, dummy_input_shape=(1, 768))
    
    try:
        print("Loading MapleONNXRunner...")
        onnx_model = MapleONNXRunner(onnx_path)
        
        # Embeddings [50, 768] (Simulating combined query+block vectors)
        dummy_input = torch.randn(50, 768)
        
        # 1. PyTorch native pass
        print("Running PyTorch Native inference...")
        with torch.no_grad():
            pt_logits = pt_model(dummy_input)
            pt_scores = torch.sigmoid(pt_logits).numpy()
            
        # 2. ONNX pass
        print("Running ONNX Native inference...")
        onnx_scores = onnx_model.predict(dummy_input.numpy())
        
        print(f"PyTorch Head Result: {pt_scores[:3]}")
        print(f"ONNX Head Result: {onnx_scores[:3]}")
        
        # 3. Validation
        print("Validating strict numerical parity (1e-5)...")
        np.testing.assert_allclose(pt_scores, onnx_scores, rtol=1e-5, atol=1e-5)
        
        print("\nSUCCESS: ONNX Runtime matches Native PyTorch output perfectly!")
        
    finally:
        print("Cleaning up models...")
        cleanup(onnx_path)

if __name__ == "__main__":
    run_test()
