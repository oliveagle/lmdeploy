// src/model/manager.mbt
// Model manager - high-level API for model loading and management
// Reference: Python core/async_engine.py, Rust model_manager.rs

use username/lmdeploy_moonbit_server/error
use username/lmdeploy_moonbit_server/config

/// Model manager - manages model loading, registry, and execution
pub struct ModelManager {
  /// Model registry
  registry: model.ModelRegistry
  /// TurboMind engine wrapper
  engine: model.TurboMindEngine
  /// Configuration
  config: config.model.ModelConfig
  /// Module-level logger
  log_enabled: Bool
}

pub fn ModelManager::new(config: config.model.ModelConfig) -> ModelManager {
  let registry = model.ModelRegistry::default()
  let engine = model.TurboMindEngine::new()
  {
    registry,
    engine,
    config,
    log_enabled: false,
  }
}

/// Initialize the model manager
pub fn ModelManager::init(self: ModelManager) -> Result[ModelManager, error.LmDeployError] {
  // Initialize engine with config
  match self.engine.init(self.config) {
    Ok(engine) => Ok({ engine, ..self }),
    Err(e) => Err(e)
  }
}

/// Load a model on startup
/// Called during server initialization
pub fn ModelManager::load_default_model(self: ModelManager) -> Result[ModelManager, error.LmDeployError] {
  let model_path = self.config.model_path()
  let model_name = self.config.model_name()

  // Skip if no model is configured
  if model_path == "" {
    log_info("No default model configured, skipping load")
    return Ok(self)
  }

  self.load_model(model_path, model_name)
}

/// Load a model by path and name
pub fn ModelManager::load_model(
  self: ModelManager,
  model_path: String,
  model_name: String,
) -> Result[ModelManager, error.LmDeployError] {
  // Check if model is already loaded
  let model_id = model.generate_model_id(model_path)
  match self.registry.get(model_id) {
    Some(m) => {
      if m.state.is_loaded() {
        log_info("Model already loaded: " ++ model_name)
        return Ok(self)
      }
    }
    None => ()
  }

  // Check capacity
  if !self.registry.can_load() {
    return Err(error::internal_error("maximum loaded models reached, please unload a model first"))
  }

  log_info("Loading model: " ++ model_name ++ " from " ++ model_path)

  // Create model metadata
  let metadata = model.ModelMetadata::from_config(self.config)
  let loaded_metadata = {
    id: model_id,
    name: model_name,
    path: model_path,
    max_context_length: self.config.max_context_length(),
    vocab_size: metadata.vocab_size,
    supports_streaming: metadata.supports_streaming,
    capabilities: metadata.capabilities,
    model_type: metadata.model_type,
    is_quantized: metadata.is_quantized,
    quantization: metadata.quantization,
  }

  // Create model entry in loading state
  let model_entry = model.LoadedModel::new(loaded_metadata)
    .with_state(model.ModelLoadState::Loading { progress: 50, stage: "loading_weights" })

  // Register the model first (shows as loading)
  let registry = self.registry.register(model_entry)

  // Actually load via engine (TODO: FFI)
  let engine_result = self.engine.load_model(model_path, model_name, self.config.max_context_length())

  match engine_result {
    Ok(engine) => {
      // Update model state to loaded
      let loaded_model = {
        id: model_id,
        metadata: loaded_metadata,
        state: model.ModelLoadState::Loaded {
          name: model_name,
          path: model_path,
          loaded_at: current_timestamp(),
          memory_mb: engine.memory_mb() / self.registry.loaded_count(),
        },
        requests_served: 0,
        tokens_generated: 0,
        loaded_at: current_timestamp(),
        memory_mb: engine.memory_mb(),
      }

      let registry = registry.update_model(model_id, loaded_model)
      log_info("Model loaded successfully: " ++ model_name)

      Ok({ registry, engine, ..self })
    }
    Err(e) => {
      // Update model state to failed
      let failed_model = {
        id: model_id,
        metadata: loaded_metadata,
        state: model.ModelLoadState::LoadFailed { error: "Failed to load: " ++ e.to_string() },
        requests_served: 0,
        tokens_generated: 0,
        loaded_at: 0L,
        memory_mb: 0,
      }

      let registry = registry.update_model(model_id, failed_model)
      log_error("Failed to load model: " ++ model_name)
      Err(e)
    }
  }
}

/// Hot-reload a model (API-triggered)
/// Reloads the specified model path with new configuration
pub fn ModelManager::hot_reload_model(
  self: ModelManager,
  model_path: String,
  model_name: String,
) -> Result[ModelManager, error.LmDeployError] {
  // Unload first if loaded
  let model_id = model.generate_model_id(model_path)
  match self.registry.get(model_id) {
    Some(m) => {
      if m.state.is_loaded() {
        self.unload_model(model_path)?
      }
    }
    None => ()
  }

  // Reload with new config
  self.load_model(model_path, model_name)
}

/// Unload a model to free memory
pub fn ModelManager::unload_model(
  self: ModelManager,
  model_path: String,
) -> Result[ModelManager, error.LmDeployError] {
  let model_id = model.generate_model_id(model_path)

  match self.registry.get(model_id) {
    Some(model) => {
      if !model.state.is_loaded() {
        return Err(error::invalid_request("model is not loaded"))
      }

      log_info("Unloading model: " ++ model.metadata.name)

      // Update state to unloading
      let unloading_model = model.with_state(model.ModelLoadState::Unloading)
      let registry = self.registry.update_model(model_id, unloading_model)

      // Actually unload via engine (TODO: FFI)
      match self.engine.unload_model(model_path) {
        Ok(engine) => {
          // Update state to not loaded
          let unloaded_model = model.with_state(model.ModelLoadState::NotLoaded)
          let registry = registry.update_model(model_id, unloaded_model)
          log_info("Model unloaded: " ++ model.metadata.name)
          Ok({ registry, engine, ..self })
        }
        Err(e) => Err(e)
      }
    }
    None => Err(error::model_not_found(model_id))
  }
}

/// Get a model by name
pub fn ModelManager::get_model(self: ModelManager, model_name: String) -> Result[model.LoadedModel, error.LmDeployError] {
  match self.registry.get_by_name(model_name) {
    Some(m) => Ok(m)
    None => Err(error::model_not_found(model_name))
  }
}

/// Get the default model
pub fn ModelManager::get_default_model(self: ModelManager) -> Result[model.LoadedModel, error.LmDeployError] {
  match self.registry.get_default() {
    Some(m) => Ok(m)
    None => Err(error::model_not_found("no default model"))
  }
}

/// List all available models
pub fn ModelManager::list_models(self: ModelManager) -> Array[model.ModelMetadata] {
  let mut models = []
  for (_, m) in self.registry.models {
    models = models.push(m.metadata)
  }
  models
}

/// List all loaded models
pub fn ModelManager::list_loaded_models(self: ModelManager) -> Array[String] {
  self.registry.loaded_models
}

/// Get loading progress for a model
pub fn ModelManager::get_load_progress(self: ModelManager, model_id: String) -> Option[model.LoadProgress] {
  match self.registry.get(model_id) {
    Some(m) => {
      match m.state {
        model.ModelLoadState::Loading { progress, stage } => {
          Some(model.LoadProgress::new(model_id).with_progress(progress, stage))
        }
        model.ModelLoadState::Loaded _ => {
          Some(model.LoadProgress::new(model_id).complete())
        }
        model.ModelLoadState::LoadFailed { error } => {
          Some(model.LoadProgress::new(model_id).failed(error))
        }
        _ => None
      }
    }
    None => None
  }
}

/// Get model info (for /v1/models endpoint)
pub fn ModelManager::get_model_info(self: ModelManager, model_name: String) -> Result[model.ModelMetadata, error.LmDeployError] {
  match self.registry.get_by_name(model_name) {
    Some(m) => Ok(m.metadata)
    None => Err(error::model_not_found(model_name))
  }
}

/// Get total memory usage
pub fn ModelManager::total_memory_mb(self: ModelManager) -> Int {
  self.engine.memory_mb()
}

/// Check if manager is ready
pub fn ModelManager::is_ready(self: ModelManager) -> Bool {
  self.engine.is_ready()
}

// Module entry point - exports all model management functionality
pub fn ModelManager::to_prometheus(self: ModelManager) -> String {
  let mut output = "# HELP lmdeploy_models_total Total number of registered models\n"
  output = output ++ "# TYPE lmdeploy_models_total gauge\n"
  output = output ++ "lmdeploy_models_total " ++ int_to_string(self.registry.list_ids().length()) ++ "\n\n"

  output = output ++ "# HELP lmdeploy_models_loaded Number of currently loaded models\n"
  output = output ++ "# TYPE lmdeploy_models_loaded gauge\n"
  output = output ++ "lmdeploy_models_loaded " ++ int_to_string(self.registry.loaded_count()) ++ "\n\n"

  output = output ++ "# HELP lmdeploy_model_memory_bytes Total GPU memory used by models\n"
  output = output ++ "# TYPE lmdeploy_model_memory_bytes gauge\n"
  output = output ++ "lmdeploy_model_memory_bytes " ++ int_to_string(self.engine.memory_mb() * 1024 * 1024) ++ "\n\n"

  output = output ++ "# HELP lmdeploy_model_load_state Model loading state (0=not_loaded, 1=loading, 2=loaded, 3=failed)\n"
  output = output ++ "# TYPE lmdeploy_model_load_state gauge\n"

  for (id, model) in self.registry.models {
    let state_value = match model.state {
      model.ModelLoadState::NotLoaded => 0
      model.ModelLoadState::Loading(_) => 1
      model.ModelLoadState::Loaded(_) => 2
      model.ModelLoadState::LoadFailed(_) => 3
      model.ModelLoadState::Unloading => 0
    }
    output = output ++ "lmdeploy_model_load_state{model_id=\"" ++ id ++ "\",model_name=\"" ++ model.metadata.name ++ "\"} " ++ int_to_string(state_value) ++ "\n"
  }

  output
}

// Helper functions

fn log_info(msg: String) -> Unit {
  // TODO: Integrate with logger module
  // For now, just ignore
  let _ = msg
}

fn log_error(msg: String) -> Unit {
  // TODO: Integrate with logger module
  let _ = msg
}

fn current_timestamp() -> Int64 {
  // TODO: Implement using FFI
  0L
}

fn int_to_string(i: Int) -> String {
  if i < 0 {
    "-" ++ int_to_string(-i)
  } else if i < 10 {
    match i {
      0 => "0"
      1 => "1"
      2 => "2"
      3 => "3"
      4 => "4"
      5 => "5"
      6 => "6"
      7 => "7"
      8 => "8"
      _ => "9"
    }
  } else if i < 100 {
    let tens = i / 10
    let ones = i - (tens * 10)
    int_to_string(tens) ++ int_to_string(ones)
  } else if i < 1000 {
    let hundreds = i / 100
    let rest = i - (hundreds * 100)
    int_to_string(hundreds) ++ int_to_string(rest)
  } else if i < 10000 {
    let thousands = i / 1000
    let rest = i - (thousands * 1000)
    int_to_string(thousands) ++ int_to_string(rest)
  } else {
    let first = i / 1000
    int_to_string(first) ++ "000"
  }
}