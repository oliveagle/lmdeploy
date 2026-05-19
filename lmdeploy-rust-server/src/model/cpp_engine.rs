//! TurboMind C++ Engine - Pure C++ inference via C API
//!
//! This engine uses the TurboMind C API directly without any Python dependency.
//! It loads weights from HuggingFace safetensors format and performs inference
//! entirely through the C++ interface.

use std::sync::Arc;
use std::time::Instant;

use crate::error::{AppError, Result};
use crate::tokenizer::LMTokenizer;
use crate::turbomind_c::{
    EngineConfig, GenConfig, ModelRequest, ScheduleMetrics, TensorMap, TurboMind,
};

/// Engine type selector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineType {
    /// Python bridge (default, compatible with all models)
    #[default]
    PythonBridge,
    /// Pure C++ inference (no Python dependency)
    PureCpp,
}

impl EngineType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "python" | "bridge" | "py" | "python_bridge" => Some(EngineType::PythonBridge),
            "cpp" | "c++" | "native" | "pure_cpp" => Some(EngineType::PureCpp),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            EngineType::PythonBridge => "python_bridge",
            EngineType::PureCpp => "pure_cpp",
        }
    }
}

/// Model loading state (same as Python bridge for compatibility)
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ModelState {
    #[default]
    Unloaded,
    Loading,
    Ready,
    Failed(String),
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub state: ModelState,
    pub loaded_at: Option<i64>,
    pub engine_type: EngineType,
    pub quant_policy: i32,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            name: "default".into(),
            path: "".into(),
            state: ModelState::Unloaded,
            loaded_at: None,
            engine_type: EngineType::default(),
            quant_policy: 0,
        }
    }
}

/// Detect AWQ quantization from config.json
fn detect_awq_quantization(model_path: &std::path::Path) -> bool {
    let config_path = model_path.join("config.json");
    if !config_path.exists() {
        return false;
    }

    std::fs::read_to_string(&config_path)
        .ok()
        .map(|content| {
            let content_lower = content.to_lowercase();
            content_lower.contains("\"quant_method\"") && content_lower.contains("\"awq\"")
        })
        .unwrap_or(false)
}

/// TurboMind pure C++ engine
pub struct TurboMindCEngine {
    pub(super) model_path: String,
    pub model_name: String,
    pub state: ModelState,
    pub loaded_at: Option<i64>,
    pub is_ready: std::sync::atomic::AtomicBool,
    pub engine_type: EngineType,

    // C API components
    tm: Option<Arc<TurboMind>>,
    request: Option<Arc<tokio::sync::Mutex<ModelRequest>>>,

    // Tokenizer for encoding/decoding
    tokenizer: Option<LMTokenizer>,

    // Configuration
    session_len: i32,
    max_batch_size: i32,
    quant_policy: i32,
}

impl TurboMindCEngine {
    /// Create a new C++ engine and initialize via C API
    pub async fn new(model_path: &str) -> Result<Self> {
        tracing::info!(model_path, "Initializing TurboMind C++ engine");

        let model_path_obj = std::path::PathBuf::from(model_path);
        let config_json = model_path_obj.join("config.json");

        if !config_json.exists() {
            tracing::error!(model_path = %model_path, "config.json not found in model path");
            return Err(AppError::ModelLoadFailed(format!(
                "config.json not found at {}",
                model_path
            )));
        }

        // Detect AWQ quantization
        let is_awq = detect_awq_quantization(&model_path_obj);
        let quant_policy = if is_awq { 4 } else { 0 };

        if is_awq {
            tracing::info!("Detected AWQ quantized model, enabling quant_policy=4");
        }

        // Load tokenizer first
        tracing::info!("Loading tokenizer...");
        let tokenizer = match LMTokenizer::from_path(model_path) {
            Ok(t) => {
                tracing::info!(vocab_size = t.vocab_size(), "Tokenizer loaded successfully");
                Some(t)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to load tokenizer from model path");
                None
            }
        };

        // Create C API engine config
        let mut engine_config = EngineConfig::new().map_err(|e| {
            AppError::ModelLoadFailed(format!("Failed to create engine config: {:?}", e))
        })?;

        // Configure engine
        // data_type is the activation dtype (kHalf), not the weight dtype.
        // AWQ weights are kUint4 but computations happen in fp16.
        engine_config.set_data_type(crate::turbomind_c::TM_DataType::TM_DATATYPE_FP16);
        engine_config.set_session_len(65536);
        engine_config.set_max_batch_size(32);
        engine_config.set_cache_block_seq_len(64);
        engine_config.set_cache_max_block_count(0.8);
        engine_config.set_enable_prefix_caching(false);
        engine_config.set_enable_metrics(true);
        engine_config.set_quant_policy(quant_policy);
        engine_config.add_device(0); // GPU 0

        // Create TurboMind instance via C API
        tracing::info!("Creating TurboMind C++ instance...");
        let tm = TurboMind::create(model_path, &mut engine_config).map_err(|e| {
            AppError::ModelLoadFailed(format!("Failed to create TurboMind: {:?}", e))
        })?;

        // Initialize from model path (builds module tree, loads weights)
        tracing::info!("Loading weights from safetensors...");
        let device_id = 0;
        let trust_remote_code = true;

        // The InitFromPath function does the full initialization:
        // 1. CreateContext
        // 2. CreateRoot
        // 3. Build ModelWeight module tree
        // 4. Load weights from safetensors
        // 5. ProcessWeights (GPU transfer)
        // 6. CreateEngine
        tm.init_from_path(device_id, model_path, trust_remote_code)
            .map_err(|e| {
                AppError::ModelLoadFailed(format!("InitFromPath failed: {:?}", e))
            })?;

        // Create inference request
        let request = ModelRequest::create(&tm).map_err(|e| {
            AppError::ModelLoadFailed(format!("Failed to create request: {:?}", e))
        })?;

        let model_name = model_path_obj
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("default")
            .to_string();

        tracing::info!("TurboMind C++ engine initialized successfully");

        Ok(Self {
            model_path: model_path.to_string(),
            model_name,
            state: ModelState::Ready,
            loaded_at: Some(unix_timestamp()),
            is_ready: std::sync::atomic::AtomicBool::new(true),
            engine_type: EngineType::PureCpp,
            tm: Some(Arc::new(tm)),
            request: Some(Arc::new(tokio::sync::Mutex::new(request))),
            tokenizer,
            session_len: 65536,
            max_batch_size: 32,
            quant_policy,
        })
    }

    /// Check if the model is ready for inference
    pub fn is_ready(&self) -> bool {
        self.is_ready.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get model metadata
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            name: self.model_name.clone(),
            path: self.model_path.clone(),
            state: self.state.clone(),
            loaded_at: self.loaded_at,
            engine_type: self.engine_type,
            quant_policy: self.quant_policy,
        }
    }

    /// Reload the model
    pub async fn reload(&mut self, new_model_path: &str) -> Result<()> {
        tracing::info!(
            old_path = %self.model_path,
            new_path = %new_model_path,
            "Reloading TurboMind C++ engine"
        );

        self.is_ready.store(false, std::sync::atomic::Ordering::Relaxed);
        self.state = ModelState::Loading;

        // Drop existing components
        self.tm = None;
        self.request = None;

        // Re-initialize
        let new_engine = Self::new(new_model_path).await?;
        self.tm = new_engine.tm;
        self.request = new_engine.request;
        self.tokenizer = new_engine.tokenizer;
        self.model_path = new_model_path.to_string();
        self.model_name = new_engine.model_name;
        self.state = ModelState::Ready;
        self.loaded_at = Some(unix_timestamp());
        self.is_ready.store(true, std::sync::atomic::Ordering::Relaxed);

        tracing::info!(
            model_path = %self.model_path,
            model_name = %self.model_name,
            "TurboMind C++ engine reloaded successfully"
        );

        Ok(())
    }

    /// Generate text with TurboMind C++ engine
    pub async fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        let (text, _, _) = self.generate_with_metrics(prompt, max_tokens).await;
        text
    }

    /// Generate text with TurboMind C++ engine, returning (text, num_tokens, elapsed_ms)
    pub async fn generate_with_metrics(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> (String, usize, f64) {
        // TODO: Add proper concurrency control
        let _tm = self.tm.as_ref().expect("TurboMind not initialized");
        let request = self.request.as_ref().expect("Request not initialized");

        // Tokenize input
        let input_ids = if let Some(tokenizer) = &self.tokenizer {
            match tokenizer.encode(prompt, false, false) {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::error!(error = %e, "Tokenization failed");
                    return (String::new(), 0, 0.0);
                }
            }
        } else {
            tracing::error!("Tokenizer not available");
            return (String::new(), 0, 0.0);
        };

        tracing::debug!(input_len = input_ids.len(), "Tokenized prompt");

        let start = Instant::now();

        // Prepare input tensors
        let mut input_tensors = TensorMap::new().unwrap();
        let input_ids_shape = [input_ids.len() as i64];
        input_tensors.set_int64("input_ids", &input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(), &input_ids_shape);
        input_tensors.set_int32("sequence_length", &[input_ids.len() as i32], &[1]);

        // Prepare generation config
        let mut gen_cfg = GenConfig::new().unwrap();
        gen_cfg.set_max_new_tokens(max_tokens as i32);
        gen_cfg.set_temperature(0.7);
        gen_cfg.set_top_p(0.95);
        gen_cfg.set_top_k(50);

        // Prepare session parameters
        use crate::turbomind_c::TM_SessionParam;
        let session = TM_SessionParam {
            id: 1,
            step: 0,
            start_flag: true,
            end_flag: true,
        };

        // Prepare output tensors
        let mut output_tensors = TensorMap::new().unwrap();

        // Run inference
        let mut req_guard = request.lock().await;
        match req_guard.forward(
            &mut input_tensors,
            &session,
            &gen_cfg,
            false, // stream_output
            true,  // enable_metrics
            &mut output_tensors,
        ) {
            Ok(_) => {
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                // Get output tokens from the request
                match req_guard.get_output("output_ids") {
                    Ok((data_ptr, size)) => {
                        // data_ptr points to int32 array
                        let num_tokens = size / 4;
                        let output_ids: Vec<i32> = unsafe {
                            std::slice::from_raw_parts(data_ptr as *const i32, num_tokens).to_vec()
                        };

                        // Decode output tokens
                        let text = if let Some(tokenizer) = &self.tokenizer {
                            match tokenizer.decode(&output_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(), true) {
                                Ok(t) => t,
                                Err(e) => {
                                    tracing::error!(error = %e, "Decoding failed");
                                    format!("[decode error: {}]", e)
                                }
                            }
                        } else {
                            format!("{:?}", output_ids)
                        };

                        (text, num_tokens, elapsed_ms)
                    }
                    Err(e) => {
                        tracing::error!(error = ?e, "Failed to get output_ids");
                        (String::new(), 0, elapsed_ms)
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = ?e, "C++ inference failed");
                (String::new(), 0, 0.0)
            }
        }
    }

    /// Generate text with streaming output (placeholder for now)
    pub async fn generate_stream(&self, prompt: &str) -> std::pin::Pin<Box<dyn futures::Stream<Item = String> + Send>> {
        // For now, fall back to non-streaming
        let text = self.generate(prompt, 512).await;
        Box::pin(futures::stream::once(async move { text }))
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> Option<&LMTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Generate embeddings (not supported by C++ engine yet)
    pub async fn embed(&self, _text: &str, _dimensions: Option<usize>) -> Vec<f32> {
        tracing::warn!("Embeddings not supported by C++ engine yet");
        Vec::new()
    }

    /// Get schedule metrics
    pub fn get_metrics(&self) -> Result<ScheduleMetrics> {
        let tm = self.tm.as_ref().expect("TurboMind not initialized");
        tm.get_schedule_metrics(0).map_err(|e| {
            AppError::InferenceFailed(format!("Failed to get metrics: {:?}", e))
        })
    }
}

fn unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_type_from_str() {
        assert_eq!(EngineType::from_str("python"), Some(EngineType::PythonBridge));
        assert_eq!(EngineType::from_str("bridge"), Some(EngineType::PythonBridge));
        assert_eq!(EngineType::from_str("py"), Some(EngineType::PythonBridge));
        assert_eq!(EngineType::from_str("cpp"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("c++"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("native"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("invalid"), None);
    }

    #[test]
    fn test_engine_type_as_str() {
        assert_eq!(EngineType::PythonBridge.as_str(), "python_bridge");
        assert_eq!(EngineType::PureCpp.as_str(), "pure_cpp");
    }

    #[test]
    fn test_detect_awq_quantization() {
        // Test with non-existent path
        let path = std::path::PathBuf::from("/nonexistent/path");
        assert!(!detect_awq_quantization(&path));
    }
}
