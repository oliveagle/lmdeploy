//! Engine abstraction layer - Pure C++ TurboMind engine
//!
//! This module provides a unified interface for C++ TurboMind inference.

mod manager;
pub mod cpp_engine;
pub mod benchmark;

// Re-export common types
pub use cpp_engine::{EngineType, TurboMindCEngine, ModelState, ModelInfo};
pub use manager::{ModelEngine, ModelManager, ModelLoadProgress, ModelLoadTracker};