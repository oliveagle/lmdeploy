//! Model Manager - Handles multiple model instances and routing
//!
//! This module provides:
//! - Dynamic model loading/unloading
//! - Model routing (multiple models support)
//! - Hot loading via API
//! - Model loading progress tracking
//!
//! NOTE: The underlying engine is a mock implementation. Model loading/unloading
//! is simulated with delays. Real TurboMind engine integration is pending.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{AppError, Result};
use crate::model::engine::{ModelState, ModelInfo, TurboMindEngine};

/// Model Manager - manages multiple model instances
pub struct ModelManager {
    /// Map of model name -> engine instance
    models: HashMap<String, Arc<RwLock<TurboMindEngine>>>,
    /// Default model name (used when no model specified)
    default_model: String,
}

impl ModelManager {
    /// Create a new ModelManager with no models loaded
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            default_model: "default".to_string(),
        }
    }

    /// Create a new ModelManager with a default model loaded
    pub async fn with_default_model(model_path: &str) -> Result<Self> {
        let mut manager = Self::new();
        let engine = TurboMindEngine::new(model_path).await?;
        let model_name = engine.model_name.clone();

        tracing::info!(
            model_name = %model_name,
            model_path = %model_path,
            "Loaded default model"
        );

        manager.models.insert(model_name.clone(), Arc::new(RwLock::new(engine)));
        manager.default_model = model_name;

        Ok(manager)
    }

    /// Load a new model
    pub async fn load_model(&mut self, model_name: &str, model_path: &str) -> Result<()> {
        if self.models.contains_key(model_name) {
            return Err(AppError::ModelAlreadyLoaded(model_name.to_string()));
        }

        tracing::info!(
            model_name = %model_name,
            model_path = %model_path,
            "Loading new model"
        );

        let engine = TurboMindEngine::new(model_path).await?;
        self.models.insert(model_name.to_string(), Arc::new(RwLock::new(engine)));

        tracing::info!(model_name = %model_name, "Model loaded successfully");
        Ok(())
    }

    /// Unload a model (release memory)
    pub async fn unload_model(&mut self, model_name: &str) -> Result<()> {
        if model_name == self.default_model {
            return Err(AppError::CannotUnloadDefaultModel);
        }

        self.models
            .remove(model_name)
            .ok_or_else(|| AppError::ModelNotFound(model_name.to_string()))?;

        tracing::info!(model_name = %model_name, "Model unloaded");
        Ok(())
    }

    /// Reload an existing model (hot reload)
    pub async fn reload_model(&mut self, model_name: &str, new_path: &str) -> Result<()> {
        let engine = self
            .models
            .get(model_name)
            .ok_or_else(|| AppError::ModelNotFound(model_name.to_string()))?;

        engine.write().await.reload(new_path).await?;

        tracing::info!(
            model_name = %model_name,
            new_path = %new_path,
            "Model reloaded successfully"
        );
        Ok(())
    }

    /// Get a model by name (or default if not specified)
    pub fn get_model(&self, model_name: Option<&str>) -> Option<Arc<RwLock<TurboMindEngine>>> {
        let name = model_name.unwrap_or(&self.default_model);
        self.models.get(name).cloned()
    }

    /// Get the default model name
    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    /// List all loaded models
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let mut infos = Vec::new();
        for engine in self.models.values() {
            let eng = engine.read().await;
            infos.push(eng.info());
        }
        infos
    }

    /// Get info for a specific model
    pub async fn get_model_info(&self, model_name: &str) -> Option<ModelInfo> {
        let engine = self.models.get(model_name)?;
        Some(engine.read().await.info())
    }

    /// Check if a model is loaded
    pub fn has_model(&self, model_name: &str) -> bool {
        self.models.contains_key(model_name)
    }

    /// Get the number of loaded models
    pub fn model_count(&self) -> usize {
        self.models.len()
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Model loading progress tracker
#[derive(Debug, Clone)]
pub struct ModelLoadProgress {
    pub model_name: String,
    pub progress: f32, // 0.0 to 1.0
    pub state: ModelState,
    pub message: String,
}

impl Default for ModelLoadProgress {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            progress: 0.0,
            state: ModelState::Unloaded,
            message: String::new(),
        }
    }
}

/// Active model loading operations
#[derive(Default)]
pub struct ModelLoadTracker {
    /// Map of model name -> loading progress
    pub loading: Arc<RwLock<HashMap<String, ModelLoadProgress>>>,
}

impl ModelLoadTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start tracking a model load
    pub async fn start_loading(&self, model_name: &str, model_path: &str) {
        let mut loading = self.loading.write().await;
        loading.insert(
            model_name.to_string(),
            ModelLoadProgress {
                model_name: model_name.to_string(),
                progress: 0.0,
                state: ModelState::Loading,
                message: format!("Loading model from {}", model_path),
            },
        );
    }

    /// Update loading progress
    pub async fn update_progress(&self, model_name: &str, progress: f32, message: &str) {
        let mut loading = self.loading.write().await;
        if let Some(entry) = loading.get_mut(model_name) {
            entry.progress = progress;
            entry.message = message.to_string();
        }
    }

    /// Mark model as loaded
    pub async fn finish_loading(&self, model_name: &str) {
        let mut loading = self.loading.write().await;
        if let Some(entry) = loading.get_mut(model_name) {
            entry.progress = 1.0;
            entry.state = ModelState::Ready;
            entry.message = "Model loaded successfully".to_string();
        }
    }

    /// Mark model as failed
    pub async fn fail_loading(&self, model_name: &str, error: &str) {
        let mut loading = self.loading.write().await;
        if let Some(entry) = loading.get_mut(model_name) {
            entry.progress = 0.0;
            entry.state = ModelState::Failed(error.to_string());
            entry.message = format!("Failed to load model: {}", error);
        }
    }

    /// Get progress for a specific model
    pub async fn get_progress(&self, model_name: &str) -> Option<ModelLoadProgress> {
        let loading = self.loading.read().await;
        loading.get(model_name).cloned()
    }

    /// Get all loading operations
    pub async fn list_loading(&self) -> Vec<ModelLoadProgress> {
        let loading = self.loading.read().await;
        loading.values().cloned().collect()
    }

    /// Clear completed loading operations
    pub async fn clear_completed(&self) {
        let mut loading = self.loading.write().await;
        loading.retain(|_, p| matches!(p.state, ModelState::Loading));
    }
}
