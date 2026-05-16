// src/model/model.mbt
// Model loading and management for LMDeploy MoonBit API
// Reference: Python core/async_engine.py, Rust model loader

use username/lmdeploy_moonbit_server/error
use username/lmdeploy_moonbit_server/config

/// Model loading state
pub enum ModelLoadState {
  /// Model is not loaded
  NotLoaded
  /// Model is currently loading
  Loading {
    /// Loading progress (0-100)
    progress: Int,
    /// Current loading stage
    stage: String,
  }
  /// Model is loaded and ready
  Loaded {
    /// Model name
    name: String,
    /// Model path
    path: String,
    /// Timestamp when model was loaded
    loaded_at: Int64,
    /// Current memory usage in MB
    memory_mb: Int,
  }
  /// Model failed to load
  LoadFailed {
    /// Error message
    error: String,
  }
  /// Model is unloading
  Unloading
}

pub fn ModelLoadState::is_loaded(self: ModelLoadState) -> Bool {
  match self {
    Loaded(_) => true
    _ => false
  }
}

pub fn ModelLoadState::is_loading(self: ModelLoadState) -> Bool {
  match self {
    Loading(_) => true
    _ => false
  }
}

pub fn ModelLoadState::to_string(self: ModelLoadState) -> String {
  match self {
    NotLoaded => "not_loaded"
    Loading { progress, stage } => "loading:" ++ int_to_string(progress) ++ ":" ++ stage
    Loaded { name, .. } => "loaded:" ++ name
    LoadFailed { error } => "failed:" ++ error
    Unloading => "unloading"
  }
}

/// Model metadata
pub struct ModelMetadata {
  /// Unique model identifier
  id: String
  /// Model name (display name)
  name: String
  /// Model path on disk
  path: String
  /// Maximum context length in tokens
  max_context_length: Int
  /// Vocabulary size
  vocab_size: Int
  /// Whether model supports streaming
  supports_streaming: Bool
  /// Model capabilities
  capabilities: Array[String]
  /// Model type (llama, qwen, etc.)
  model_type: String
  /// Whether model is quantized
  is_quantized: Bool
  /// Quantization type (awq, gptq, etc.) or "none"
  quantization: String
}

pub fn ModelMetadata::default() -> ModelMetadata {
  {
    id: "",
    name: "default",
    path: "",
    max_context_length: 8192,
    vocab_size: 152064,
    supports_streaming: true,
    capabilities: ["chat", "completion"],
    model_type: "unknown",
    is_quantized: false,
    quantization: "none",
  }
}

pub fn ModelMetadata::from_config(cfg: config.model.ModelConfig) -> ModelMetadata {
  {
    id: generate_model_id(cfg.model_path()),
    name: cfg.model_name(),
    path: cfg.model_path(),
    max_context_length: cfg.max_context_length(),
    vocab_size: 152064,  // Default, would be detected from model
    supports_streaming: true,
    capabilities: ["chat", "completion", "embeddings"],
    model_type: detect_model_type(cfg.model_path()),
    is_quantized: is_path_quantized(cfg.model_path()),
    quantization: detect_quantization(cfg.model_path()),
  }
}

/// A loaded model instance
pub struct LoadedModel {
  /// Model metadata
  metadata: ModelMetadata
  /// Current load state
  state: ModelLoadState
  /// Number of requests served by this model
  requests_served: Int
  /// Total tokens generated
  tokens_generated: Int
  /// Load timestamp
  loaded_at: Int64
  /// Estimated memory usage in MB
  memory_mb: Int
}

pub fn LoadedModel::new(metadata: ModelMetadata) -> LoadedModel {
  {
    metadata,
    state: NotLoaded,
    requests_served: 0,
    tokens_generated: 0,
    loaded_at: 0L,
    memory_mb: 0,
  }
}

pub fn LoadedModel::with_state(self: LoadedModel, state: ModelLoadState) -> LoadedModel {
  { state, ..self }
}

pub fn LoadedModel::with_loaded(self: LoadedModel, timestamp: Int64, memory_mb: Int) -> LoadedModel {
  {
    loaded_at: timestamp,
    memory_mb,
    state: Loaded {
      name: self.metadata.name,
      path: self.metadata.path,
      loaded_at: timestamp,
      memory_mb,
    },
    ..self
  }
}

pub fn LoadedModel::record_request(self: LoadedModel, tokens: Int) -> LoadedModel {
  {
    requests_served: self.requests_served + 1,
    tokens_generated: self.tokens_generated + tokens,
    ..self
  }
}

/// Model registry for managing multiple models
pub struct ModelRegistry {
  /// All registered models (id -> model)
  models: Map[String, LoadedModel]
  /// Default model ID
  default_model_id: String
  /// Maximum number of models that can be loaded simultaneously
  max_loaded_models: Int
  /// Currently loaded model IDs
  loaded_models: Array[String]
}

pub fn ModelRegistry::new(max_loaded: Int) -> ModelRegistry {
  {
    models: Map::new(),
    default_model_id: "",
    max_loaded_models: max_loaded,
    loaded_models: [],
  }
}

pub fn ModelRegistry::default() -> ModelRegistry {
  ModelRegistry::new(3)  // Default: up to 3 models loaded simultaneously
}

/// Register a model in the registry
pub fn ModelRegistry::register(self: ModelRegistry, model: LoadedModel) -> ModelRegistry {
  let id = model.metadata.id
  {
    models: self.models.insert(id, model),
    default_model_id: if self.default_model_id == "" { id } else { self.default_model_id },
    ..self
  }
}

/// Get a model by ID
pub fn ModelRegistry::get(self: ModelRegistry, id: String) -> Option[LoadedModel] {
  self.models.get(id)
}

/// Get a model by name (searches all models)
pub fn ModelRegistry::get_by_name(self: ModelRegistry, name: String) -> Option[LoadedModel] {
  for (_, model) in self.models {
    if model.metadata.name == name {
      return Some(model)
    }
  }
  None
}

/// Get the default model
pub fn ModelRegistry::get_default(self: ModelRegistry) -> Option[LoadedModel] {
  if self.default_model_id == "" {
    None
  } else {
    self.models.get(self.default_model_id)
  }
}

/// Set the default model
pub fn ModelRegistry::set_default(self: ModelRegistry, id: String) -> Result[ModelRegistry, error.LmDeployError] {
  match self.models.get(id) {
    Some(_) => Ok({ default_model_id: id, ..self })
    None => Err(error::model_not_found(id))
  }
}

/// List all registered model IDs
pub fn ModelRegistry::list_ids(self: ModelRegistry) -> Array[String] {
  let mut ids = []
  for (id, _) in self.models {
    ids = ids.concat([id])
  }
  ids
}

/// List all loaded model IDs
pub fn ModelRegistry::list_loaded(self: ModelRegistry) -> Array[String] {
  self.loaded_models
}

/// Get number of currently loaded models
pub fn ModelRegistry::loaded_count(self: ModelRegistry) -> Int {
  self.loaded_models.length()
}

/// Check if a model can be loaded (has capacity)
pub fn ModelRegistry::can_load(self: ModelRegistry) -> Bool {
  self.loaded_models.length() < self.max_loaded_models
}

/// Update model state after load/unload operations
pub fn ModelRegistry::update_model(self: ModelRegistry, id: String, model: LoadedModel) -> ModelRegistry {
  let was_loaded = match self.models.get(id) {
    Some(m) => m.state.is_loaded()
    None => false
  }
  let is_loaded = model.state.is_loaded()

  let loaded_models = if was_loaded && !is_loaded {
    // Model was unloaded
    self.loaded_models.filter(fn(id) { id != model.metadata.id })
  } else if !was_loaded && is_loaded {
    // Model was loaded
    self.loaded_models.concat([model.metadata.id])
  } else {
    self.loaded_models
  }

  {
    models: self.models.insert(id, model),
    loaded_models,
    ..self
  }
}

/// Model loading progress information
pub struct LoadProgress {
  /// Model ID being loaded
  model_id: String
  /// Progress percentage (0-100)
  progress: Int
  /// Current stage description
  stage: String
  /// Whether loading is complete
  is_complete: Bool
  /// Error message if loading failed
  error: Option[String]
}

pub fn LoadProgress::new(model_id: String) -> LoadProgress {
  {
    model_id,
    progress: 0,
    stage: "initializing",
    is_complete: false,
    error: None,
  }
}

pub fn LoadProgress::with_progress(self: LoadProgress, progress: Int, stage: String) -> LoadProgress {
  { progress, stage, ..self }
}

pub fn LoadProgress::complete(self: LoadProgress) -> LoadProgress {
  { progress: 100, stage: "complete", is_complete: true, ..self }
}

pub fn LoadProgress::failed(self: LoadProgress, error: String) -> LoadProgress {
  { progress: 0, stage: "failed", is_complete: true, error: Some(error), ..self }
}

pub fn LoadProgress::to_json(self: LoadProgress) -> String {
  let error_str = match self.error {
    Some(e) => "\"" ++ e ++ "\""
    None => "null"
  }
  "{\"model_id\":\"" ++ self.model_id ++ "\",\"progress\":" ++ int_to_string(self.progress) ++ ",\"stage\":\"" ++ self.stage ++ "\",\"is_complete\":" ++ bool_to_string(self.is_complete) ++ ",\"error\":" ++ error_str ++ "}"
}

// Helper functions

/// Generate a unique model ID from path
pub fn generate_model_id(path: String) -> String {
  // Simple hash-based ID generation
  let hash = path.length() * 31
  "model-" ++ int_to_string(hash)
}

/// Detect model type from path
fn detect_model_type(path: String) -> String {
  if path.contains("qwen") {
    "qwen"
  } else if path.contains("llama") {
    "llama"
  } else if path.contains("internlm") {
    "internlm"
  } else {
    "unknown"
  }
}

/// Check if model path indicates quantization
fn is_path_quantized(path: String) -> Bool {
  path.contains("awq") || path.contains("gptq") || path.contains("gguf")
}

/// Detect quantization type from path
fn detect_quantization(path: String) -> String {
  if path.contains("awq") {
    "awq"
  } else if path.contains("gptq") {
    "gptq"
  } else if path.contains("gguf") {
    "gguf"
  } else if path.contains("w8a8") {
    "w8a8"
  } else {
    "none"
  }
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

fn bool_to_string(b: Bool) -> String {
  if b { "true" } else { "false" }
}
