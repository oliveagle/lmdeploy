//! Engine abstraction layer - supports both PythonBridge and Pure C++ engines
//!
//! This module provides a unified interface for model inference, allowing
//! the server to switch between Python Bridge (compatibility) and Pure C++
//! (no Python dependency) engines based on configuration.

mod engine;
pub mod cpp_engine;
mod manager;
mod python_bridge;
pub mod benchmark;

// Re-export common types
pub use cpp_engine::{EngineType, TurboMindCEngine};
pub use engine::{ModelState, ModelInfo, TurboMindEngine};
pub use manager::{ModelEngine, ModelManager, ModelLoadProgress, ModelLoadTracker};
pub use python_bridge::PythonBridge;
