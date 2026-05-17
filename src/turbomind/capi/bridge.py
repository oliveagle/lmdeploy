#!/usr/bin/env python3
"""Bridge: Load LMDeploy TurboMind model and expose C pointer via ctypes."""
import ctypes
import ctypes.util
import sys
import os

# Add parent directory to path to import lmdeploy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lmdeploy.turbomind import TurboMind

# Global model instance
_model_instance = None
_model_ptr = 0

# Create a function that can be called from C
# This stores the model and returns its pointer

class ModelHandle(ctypes.Structure):
    _fields_ = [
        ("tm_ptr", ctypes.c_void_p),
        ("tokenizer_ptr", ctypes.c_void_p),
        ("vocab_size", ctypes.c_int),
        ("hidden_units", ctypes.c_int),
        ("session_len", ctypes.c_int),
    ]

def init_model(model_path: str, session_len: int = 2048):
    """Load model and return handle."""
    global _model_instance

    engine_config = TurboMind.get_engine_config(
        model_path=model_path,
        session_len=session_len,
    )

    tm = TurboMind.from_pretrained(
        model_path=model_path,
        engine_config=engine_config,
    )

    _model_instance = tm

    handle = ModelHandle()
    handle.tm_ptr = 0  # Can't get C pointer directly
    handle.vocab_size = tm._vocab_size
    handle.session_len = session_len
    return handle

if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B'
    print(f"Loading model from {model_path}...")
    handle = init_model(model_path)
    print(f"Model loaded: vocab_size={handle.vocab_size}, session_len={handle.session_len}")
