//! Model Manager - Handles multiple model instances and routing
//!
//! This module provides:
//! - Dynamic model loading/unloading
//! - Model routing (multiple models support)
//! - Hot loading via API
//! - Model loading progress tracking
//! - Pure C++ engine only

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{AppError, Result};
use crate::model::cpp_engine::{EngineType, ModelState, ModelInfo, TurboMindCEngine, GenerationParams, TokenLogprob};
use crate::tokenizer::LMTokenizer;

/// Unified engine enum - Pure C++ only
pub enum ModelEngine {
    /// Pure C++ engine (no Python dependency)
    PureCpp(TurboMindCEngine),
}

impl ModelEngine {
    pub fn engine_type(&self) -> EngineType {
        match self {
            ModelEngine::PureCpp(_) => EngineType::PureCpp,
        }
    }

    pub async fn generate(&self, prompt: &str, params: GenerationParams) -> String {
        match self {
            ModelEngine::PureCpp(e) => e.generate(prompt, params).await,
        }
    }

    pub async fn generate_with_metrics(&self, prompt: &str, params: GenerationParams) -> (String, usize, f64) {
        match self {
            ModelEngine::PureCpp(e) => e.generate_with_metrics(prompt, params).await,
        }
    }

    pub async fn generate_with_logprobs(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> (String, usize, f64, Option<Vec<TokenLogprob>>) {
        match self {
            ModelEngine::PureCpp(e) => e.generate_with_logprobs(prompt, params).await,
        }
    }

    pub async fn generate_stream(&self, prompt: &str, params: GenerationParams) -> std::pin::Pin<Box<dyn futures::Stream<Item = String> + Send>> {
        match self {
            ModelEngine::PureCpp(e) => e.generate_stream(prompt, params).await,
        }
    }

    pub async fn embed(&self, text: &str, dimensions: Option<usize>) -> Vec<f32> {
        match self {
            ModelEngine::PureCpp(e) => e.embed(text, dimensions).await,
        }
    }

    pub async fn reload(&mut self, new_model_path: &str) -> Result<()> {
        match self {
            ModelEngine::PureCpp(e) => e.reload(new_model_path).await,
        }
    }

    pub fn info(&self) -> ModelInfo {
        match self {
            ModelEngine::PureCpp(e) => e.info(),
        }
    }

    pub fn tokenizer(&self) -> Option<&LMTokenizer> {
        match self {
            ModelEngine::PureCpp(e) => e.tokenizer(),
        }
    }
}

/// Model Manager - manages multiple model instances
pub struct ModelManager {
    /// Map of model name -> engine instance
    models: HashMap<String, Arc<RwLock<ModelEngine>>>,
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
    /// Uses the PureCpp engine type
    pub async fn with_default_model_and_type(model_path: &str, engine_type: EngineType) -> Result<Self> {
        let mut manager = Self::new();
        let model_engine = match engine_type {
            EngineType::PureCpp => {
                let cpp_engine = TurboMindCEngine::new(model_path).await?;
                ModelEngine::PureCpp(cpp_engine)
            }
        };

        let model_name = match &model_engine {
            ModelEngine::PureCpp(e) => e.model_name.clone(),
        };

        tracing::info!(
            model_name = %model_name,
            model_path = %model_path,
            engine_type = %engine_type.as_str(),
            "Loaded default model"
        );

        manager.models.insert(model_name.clone(), Arc::new(RwLock::new(model_engine)));
        manager.default_model = model_name;

        Ok(manager)
    }

    /// Load a new model with specified engine type
    pub async fn load_model_with_type(&mut self, model_name: &str, model_path: &str, engine_type: EngineType) -> Result<()> {
        if self.models.contains_key(model_name) {
            return Err(AppError::ModelAlreadyLoaded(model_name.to_string()));
        }

        tracing::info!(
            model_name = %model_name,
            model_path = %model_path,
            engine_type = %engine_type.as_str(),
            "Loading new model"
        );

        let model_engine = match engine_type {
            EngineType::PureCpp => {
                let cpp_engine = TurboMindCEngine::new(model_path).await?;
                ModelEngine::PureCpp(cpp_engine)
            }
        };

        self.models.insert(model_name.to_string(), Arc::new(RwLock::new(model_engine)));
        tracing::info!(model_name = %model_name, "Model loaded successfully");
        Ok(())
    }

    /// Load a new model (defaults to PureCpp engine)
    pub async fn load_model(&mut self, model_name: &str, model_path: &str) -> Result<()> {
        self.load_model_with_type(model_name, model_path, EngineType::PureCpp).await
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

        let mut eng = engine.write().await;
        eng.reload(new_path).await?;

        tracing::info!(
            model_name = %model_name,
            new_path = %new_path,
            "Model reloaded successfully"
        );
        Ok(())
    }

    /// Get a model by name (or default if not specified)
    pub fn get_model(&self, model_name: Option<&str>) -> Option<Arc<RwLock<ModelEngine>>> {
        let name = model_name.unwrap_or(&self.default_model);
        // Try exact match first
        if let Some(engine) = self.models.get(name) {
            return Some(engine.clone());
        }
        // If requested name is "default" or matches default_model, return the default model
        if name == "default" || name == self.default_model {
            return self.models.get(&self.default_model).cloned();
        }
        // Try matching by model path suffix
        for (key, engine) in &self.models {
            if key.contains(name) || name.contains(key) {
                return Some(engine.clone());
            }
        }
        None
    }

    /// Get the default model name
    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    /// List all loaded models
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let mut infos = Vec::new();
        for (name, engine) in self.models.iter() {
            let eng = engine.read().await;
            let info = match &*eng {
                ModelEngine::PureCpp(e) => {
                    let i = e.info();
                    ModelInfo {
                        name: name.clone(),
                        path: i.path,
                        state: i.state,
                        loaded_at: i.loaded_at,
                        engine_type: i.engine_type,
                        quant_policy: i.quant_policy,
                        hidden_size: i.hidden_size,
                    }
                }
            };
            infos.push(info);
        }
        infos
    }

    /// Get info for a specific model
    pub async fn get_model_info(&self, model_name: &str) -> Option<ModelInfo> {
        let engine = self.models.get(model_name)?;
        let eng = engine.read().await;
        let info = match &*eng {
            ModelEngine::PureCpp(e) => {
                let i = e.info();
                ModelInfo {
                    name: model_name.to_string(),
                    path: i.path,
                    state: i.state,
                    loaded_at: i.loaded_at,
                    engine_type: i.engine_type,
                    quant_policy: i.quant_policy,
                    hidden_size: i.hidden_size,
                }
            }
        };
        Some(info)
    }

    /// Check if a model is loaded
    pub fn has_model(&self, model_name: &str) -> bool {
        self.models.contains_key(model_name)
    }

    /// Get the number of loaded models
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Get tokenizer for a model by name (or default if not specified)
    pub async fn get_model_tokenizer(&self, model_name: Option<&str>) -> Option<LMTokenizer> {
        let engine = self.get_model(model_name)?;
        let eng = engine.read().await;
        match &*eng {
            ModelEngine::PureCpp(e) => {
                e.tokenizer().cloned()
            }
        }
    }

    /// Get the default model's tokenizer
    pub async fn get_default_tokenizer(&self) -> Option<LMTokenizer> {
        self.get_model_tokenizer(Some(&self.default_model)).await
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
