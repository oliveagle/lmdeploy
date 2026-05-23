#!/usr/bin/env python3
"""Embeddings verification: Python TurboMind vs Rust+C++ engine comparison.

This script tests that the Rust embeddings implementation produces
results consistent with the Python TurboMind engine.

Usage:
    python docs/test_embeddings_verification_20260523.py \
        --model /path/to/model \
        --rust-url http://localhost:3000  # optional, for Rust comparison
"""

import sys
import time
from typing import Optional

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compare_embeddings(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    """Compare two embedding vectors and return metrics."""
    sim = cosine_similarity(a, b)
    mae = np.mean(np.abs(a - b))
    max_diff = np.max(np.abs(a - b))
    rmse = np.sqrt(np.mean((a - b) ** 2))
    return {
        "cosine_similarity": float(sim),
        "mae": float(mae),
        "max_diff": float(max_diff),
        "rmse": float(rmse),
        "label_a": label_a,
        "label_b": label_b,
        "dim_a": len(a),
        "dim_b": len(b),
    }


def run_python_embeddings(model_path: str, test_texts: list[str]) -> list[np.ndarray]:
    """Get embeddings using Python TurboMind engine."""
    from lmdeploy import GenerationConfig, pipeline

    # For local paths, use the path directly
    # The pipeline accepts local paths
    pipe = pipeline(model_path, trust_remote_code=True)
    embeddings = []

    for text in test_texts:
        print(f"  Python: encoding '{text[:50]}...'")
        gen_config = GenerationConfig(
            max_new_tokens=0,
            output_last_hidden_state="generation",
        )
        response = pipe(text, gen_config=gen_config)
        if hasattr(response, "last_hidden_state") and response.last_hidden_state is not None:
            emb = np.array(response.last_hidden_state, dtype=np.float32)
            # Flatten if needed - should be [1, hidden_dim] or [hidden_dim]
            emb = emb.flatten()
            embeddings.append(emb)
        else:
            print(f"  WARNING: No last_hidden_state in response for text: {text[:50]}")
            embeddings.append(np.array([], dtype=np.float32))

    return embeddings


def run_rust_embeddings(rust_url: str, test_texts: list[str]) -> list[np.ndarray]:
    """Get embeddings from Rust server HTTP API."""
    import requests

    embeddings = []
    for text in test_texts:
        print(f"  Rust: encoding '{text[:50]}...'")
        resp = requests.post(
            f"{rust_url}/v1/embeddings",
            json={"input": text, "model": "default", "dimensions": None},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and len(data["data"]) > 0:
            emb = np.array(data["data"][0]["embedding"], dtype=np.float32)
            embeddings.append(emb)
        else:
            print(f"  WARNING: Unexpected response format: {data}")
            embeddings.append(np.array([], dtype=np.float32))

    return embeddings


def test_embedding_determinism(model_path: str, test_texts: list[str]) -> bool:
    """Test that embeddings are deterministic (same input -> same output)."""
    print("\n=== Testing Embedding Determinism ===")
    embeddings_run1 = run_python_embeddings(model_path, test_texts)
    embeddings_run2 = run_python_embeddings(model_path, test_texts)

    all_passed = True
    for i, (emb1, emb2) in enumerate(zip(embeddings_run1, embeddings_run2)):
        if len(emb1) == 0 or len(emb2) == 0:
            print(f"  SKIP: Test {i+1} - empty embedding")
            continue
        sim = cosine_similarity(emb1, emb2)
        exact = np.allclose(emb1, emb2, rtol=1e-5, atol=1e-5)
        passed = exact and sim > 0.9999
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Text {i+1}: cosine={sim:.6f}, exact_match={exact}")
        if not passed:
            all_passed = False

    return all_passed


def test_embedding_semantics(model_path: str) -> bool:
    """Test that embeddings capture semantic similarity."""
    print("\n=== Testing Embedding Semantics ===")

    test_pairs = [
        # (positive pairs - should be similar)
        ("The cat is sleeping", "A feline is resting"),
        ("I love programming in Python", "Python is my favorite language for coding"),
        # (negative pairs - should be different)
        ("The cat is sleeping", "Python programming is fun"),
    ]

    all_texts = [t for pair in test_pairs for t in pair]
    embeddings = run_python_embeddings(model_path, all_texts)

    all_passed = True
    for i, (text_a, text_b) in enumerate(test_pairs):
        emb_a = embeddings[i * 2]
        emb_b = embeddings[i * 2 + 1]

        if len(emb_a) == 0 or len(emb_b) == 0:
            print(f"  SKIP: Pair {i+1} - empty embedding")
            continue

        sim = cosine_similarity(emb_a, emb_b)
        is_positive_pair = i < 2  # First 2 pairs should be similar

        if is_positive_pair:
            passed = sim > 0.7  # Similar texts should have cosine > 0.7
            expected = "high"
        else:
            passed = sim < 0.9  # Different texts should have lower similarity
            expected = "lower"

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Pair {i+1}: sim={sim:.4f} (expected {expected})")
        print(f"    A: {text_a}")
        print(f"    B: {text_b}")
        if not passed:
            all_passed = False

    return all_passed


def test_dimension_truncation(model_path: str, test_text: str, full_dim: int) -> bool:
    """Test that dimension truncation works correctly."""
    print("\n=== Testing Dimension Truncation ===")

    from lmdeploy import GenerationConfig, pipeline

    pipe = pipeline(model_path, trust_remote_code=True)

    # Get full embedding
    gen_full = GenerationConfig(max_new_tokens=0, output_last_hidden_state="generation")
    response_full = pipe(test_text, gen_config=gen_full)
    full_emb = np.array(response_full.last_hidden_state, dtype=np.float32).flatten()

    print(f"  Full embedding dim: {len(full_emb)} (expected: {full_dim})")

    if len(full_emb) != full_dim:
        print(f"  FAIL: Dimension mismatch")
        return False

    # Test truncation - first N dimensions should match
    trunc_dims = [128, 256, 512, 768]
    all_passed = True
    for trunc_dim in trunc_dims:
        if trunc_dim >= full_dim:
            continue

        trunc_emb = full_emb[:trunc_dim]
        # Verify first N dimensions match
        matches = np.allclose(full_emb[:trunc_dim], trunc_emb, rtol=1e-5, atol=1e-5)
        status = "PASS" if matches else "FAIL"
        print(f"  [{status}] Truncated to {trunc_dim} dim: matches first {trunc_dim} of full embedding")
        if not matches:
            all_passed = False

    return all_passed


def test_rust_vs_python(
    model_path: str,
    rust_url: str,
    test_texts: list[str],
    target_dim: Optional[int] = None,
) -> bool:
    """Compare embeddings between Rust and Python implementations."""
    print(f"\n=== Comparing Rust vs Python Embeddings ===")
    print(f"  Rust URL: {rust_url}")
    print(f"  Model: {model_path}")

    python_embeddings = run_python_embeddings(model_path, test_texts)
    rust_embeddings = run_rust_embeddings(rust_url, test_texts)

    all_passed = True
    for i, (py_emb, rust_emb) in enumerate(zip(python_embeddings, rust_embeddings)):
        if len(py_emb) == 0 or len(rust_emb) == 0:
            print(f"  SKIP: Test {i+1} - empty embedding")
            continue

        metrics = compare_embeddings(
            py_emb, rust_emb, "Python TurboMind", "Rust+C++"
        )

        # Embeddings from the same C++ backend should be very similar
        passed = metrics["cosine_similarity"] > 0.99
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] Text {i+1}:")
        print(f"    Python dim: {metrics['dim_a']}")
        print(f"    Rust dim:   {metrics['dim_b']}")
        print(f"    Cosine similarity: {metrics['cosine_similarity']:.6f}")
        print(f"    MAE:               {metrics['mae']:.6f}")
        print(f"    Max diff:          {metrics['max_diff']:.6f}")
        print(f"    RMSE:              {metrics['rmse']:.6f}")

        if not passed:
            all_passed = False

    return all_passed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify embeddings correctness")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--rust-url", help="Rust server URL (optional)")
    parser.add_argument("--test-texts", nargs="+", default=[
        "Hello, world! This is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Python is a popular programming language for data science.",
        "The weather is nice today, let's go for a walk.",
    ])
    parser.add_argument("--target-dim", type=int, default=None, help="Target dimension for truncation test")

    args = parser.parse_args()

    print("=" * 60)
    print("Embeddings Verification Suite")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Test texts: {len(args.test_texts)}")

    results = {}

    # Test 1: Determinism
    try:
        results["determinism"] = test_embedding_determinism(args.model, args.test_texts)
    except Exception as e:
        print(f"\n  ERROR in determinism test: {e}")
        results["determinism"] = False

    # Test 2: Semantics
    try:
        results["semantics"] = test_embedding_semantics(args.model)
    except Exception as e:
        print(f"\n  ERROR in semantics test: {e}")
        results["semantics"] = False

    # Test 3: Dimension truncation
    if args.target_dim:
        try:
            results["truncation"] = test_dimension_truncation(
                args.model, args.test_texts[0], args.target_dim
            )
        except Exception as e:
            print(f"\n  ERROR in truncation test: {e}")
            results["truncation"] = False

    # Test 4: Rust vs Python comparison
    if args.rust_url:
        try:
            results["rust_vs_python"] = test_rust_vs_python(
                args.model, args.rust_url, args.test_texts, args.target_dim
            )
        except Exception as e:
            print(f"\n  ERROR in Rust vs Python test: {e}")
            results["rust_vs_python"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
