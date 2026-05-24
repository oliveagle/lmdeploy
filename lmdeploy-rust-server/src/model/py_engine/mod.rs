//! Python Bridge Engine - Uses Python subprocess for inference
//!
//! This engine spawns a Python process and communicates via stdin/stdout JSON.
//! It provides an alternative to the pure C++ engine for comparison.

mod process;

pub use process::PyBridgeEngine;
