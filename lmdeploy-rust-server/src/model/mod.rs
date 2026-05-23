//! Engine abstraction layer - Pure C++ TurboMind engine
//!
//! This module provides a unified interface for C++ TurboMind inference.

pub mod benchmark;
pub mod cpp_engine;
mod manager;

// Re-export common types
pub use cpp_engine::{
    BatchItem, BatchResult, EngineType, GenerationParams, ModelInfo, ModelState, TokenLogprob,
    TopLogprob, TurboMindCEngine,
};
pub use manager::{ModelEngine, ModelLoadProgress, ModelLoadTracker, ModelManager};
