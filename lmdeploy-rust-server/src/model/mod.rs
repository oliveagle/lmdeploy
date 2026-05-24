//! Engine abstraction layer - Multi-engine support for LMDeploy
//!
//! This module provides unified interfaces for multiple inference engines:
//! - Pure C++ TurboMind engine (via C API)
//! - Python bridge engine (via Python subprocess + C++ TurboMind)

pub mod benchmark;
pub mod cpp_engine;
mod manager;
pub mod python_bridge;
#[cfg(test)]
mod stress_test;

// Re-export common types
pub use cpp_engine::{
    BatchItem, BatchResult, EngineType, GenerationParams, GuidedGrammar, ModelInfo, ModelState,
    TokenLogprob, TopLogprob, TurboMindCEngine,
};
pub use manager::{ModelEngine, ModelLoadProgress, ModelLoadTracker, ModelManager};
pub use python_bridge::{PythonBridge, ScheduleMetrics};
