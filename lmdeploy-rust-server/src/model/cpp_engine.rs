//! TurboMind C++ Engine - Pure C++ inference via C API
//!
//! This engine uses the TurboMind C API directly without any Python dependency.
//! It loads weights from HuggingFace safetensors format and performs inference
//! entirely through the C++ interface.
//!
//! ## Concurrency Model
//!
//! The C++ TurboMind engine has an internal request queue (Gateway) that handles
//! concurrent scheduling. This wrapper creates a pool of ModelRequest instances
//! to allow parallel inference without mutex contention. Each request is
//! independent and can run concurrently with others.

use std::sync::Arc;
use std::time::Instant;

use crate::error::{AppError, Result};
use crate::tokenizer::LMTokenizer;
use crate::turbomind_c::{
    c_int, c_void, EngineConfig, GenConfig, ModelRequest, ScheduleMetrics, TensorMap,
    TM_SessionParam, TurboMind,
};
use serde::Serialize;

/// A single token logprobs entry as returned by the OpenAI API.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f64,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}

/// One of the top logprobs for a token position.
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f64,
    pub bytes: Vec<u8>,
}

/// Generation parameters from HTTP/gRPC requests.
///
/// These parameters are applied to the GenConfig when calling the C++ engine.
#[derive(Debug, Clone, Default)]
pub struct GenerationParams {
    /// Maximum number of new tokens to generate.
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 = deterministic, > 0.0 = random sampling).
    /// Default: 0.7
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold (0.0-1.0). Default: 0.95
    pub top_p: Option<f32>,
    /// Top-k sampling (0 = disabled). Default: 50
    pub top_k: Option<i32>,
    /// Minimum probability threshold for MinP sampling. Default: 0.0
    pub min_p: Option<f32>,
    /// Repetition penalty. Default: 1.0
    pub repetition_penalty: Option<f32>,
    /// Random seed for deterministic sampling. None = random.
    pub seed: Option<u64>,
    /// Stop sequences (not directly used by C++, handled by caller).
    pub stop: Option<Vec<String>>,
    /// Return log probabilities for each token. Default: false
    pub logprobs: Option<bool>,
    /// Number of top log probabilities to return per token. Default: 0 (none)
    pub top_logprobs: Option<u32>,
}

impl GenerationParams {
    /// Create GenerationParams from ChatCompletionsRequest fields.
    pub fn from_chat_request(
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<i32>,
        seed: Option<i32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        stop: Option<crate::handlers::http::Stop>,
        logprobs: Option<bool>,
        top_logprobs: Option<u32>,
    ) -> Self {
        let repetition_penalty = Self::compute_repetition_penalty(presence_penalty, frequency_penalty);
        Self {
            max_tokens: max_tokens.map(|t| t as usize),
            temperature,
            top_p,
            top_k: None,
            min_p: None,
            repetition_penalty,
            seed: seed.map(|s| s as u64),
            stop: stop.map(|s| match s {
                crate::handlers::http::Stop::Single(s) => vec![s],
                crate::handlers::http::Stop::Multiple(v) => v,
            }),
            logprobs,
            top_logprobs,
        }
    }

    /// Compute repetition penalty from presence/frequency penalties.
    fn compute_repetition_penalty(
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
    ) -> Option<f32> {
        let presence = presence_penalty.unwrap_or(0.0);
        let frequency = frequency_penalty.unwrap_or(0.0);
        let combined = presence + frequency;
        if combined == 0.0 {
            None
        } else {
            Some((1.0 + combined).max(0.5).min(3.0))
        }
    }

    /// Apply parameters to a GenConfig instance.
    /// Only sets values that are Some(), leaving defaults for None.
    pub fn apply_to_gen_config(&self, gen_cfg: &mut GenConfig) {
        if let Some(max_tokens) = self.max_tokens {
            gen_cfg.set_max_new_tokens(max_tokens as c_int);
        }
        if let Some(temperature) = self.temperature {
            gen_cfg.set_temperature(temperature);
        }
        if let Some(top_p) = self.top_p {
            gen_cfg.set_top_p(top_p);
        }
        if let Some(top_k) = self.top_k {
            gen_cfg.set_top_k(top_k as c_int);
        }
        if let Some(min_p) = self.min_p {
            gen_cfg.set_min_p(min_p);
        }
        if let Some(repetition_penalty) = self.repetition_penalty {
            gen_cfg.set_repetition_penalty(repetition_penalty);
        }
        if let Some(seed) = self.seed {
            gen_cfg.set_random_seed(seed);
        }
        if let Some(logprobs) = self.logprobs {
            let num_logprobs = if logprobs {
                self.top_logprobs.unwrap_or(1) as c_int
            } else {
                0
            };
            gen_cfg.set_output_logprobs(num_logprobs);
        }
    }
}

/// Token callback for event-driven streaming.
///
/// Invoked by the C++ engine whenever a new token is generated.
/// Decodes the token and sends it through the channel.
extern "C" fn token_callback(token_id: c_int, _seq_len: c_int, user_data: *mut c_void) {
    unsafe {
        let ctx = &*(user_data as *const StreamContext);

        let token_ids: Vec<u32> = vec![token_id as u32];
        let token_str = match ctx.tokenizer.decode(&token_ids, true) {
            Ok(s) if !s.is_empty() => s,
            _ => return,
        };

        let _ = ctx.tx.try_send(token_str);
    }
}

/// Default number of concurrent inference requests
/// This matches the C++ engine's internal queue capacity
const DEFAULT_CONCURRENCY: usize = 8;

/// Shared context passed to the C token callback via raw pointer.
struct StreamContext {
    tokenizer: LMTokenizer,
    tx: tokio::sync::mpsc::Sender<String>,
}

/// Pool of ModelRequest instances for concurrent inference.
///
/// Uses a `tokio::sync::Semaphore` to limit concurrent inference requests
/// and `tokio::sync::Mutex` per slot so that the async runtime can yield
/// during blocking FFI calls instead of parking OS threads.
struct RequestPool {
    /// Per-slot tokio mutex - yields during FFI calls
    slots: Vec<tokio::sync::Mutex<ModelRequest>>,
    /// Semaphore limits concurrent inference requests
    semaphore: tokio::sync::Semaphore,
}

impl RequestPool {
    /// Create a new request pool with the given concurrency level.
    fn new(tm: &TurboMind, concurrency: usize) -> Result<Self> {
        let mut slots = Vec::with_capacity(concurrency);
        for i in 0..concurrency {
            let request = ModelRequest::create(tm).map_err(|e| {
                AppError::ModelLoadFailed(format!("Failed to create request #{}: {:?}", i, e))
            })?;
            slots.push(tokio::sync::Mutex::new(request));
        }
        Ok(Self {
            slots,
            semaphore: tokio::sync::Semaphore::new(concurrency),
        })
    }

    /// Acquire a slot (async). Returns a guard that holds the semaphore
    /// permit and the mutex guard. The slot is released when the guard is
    /// dropped.
    ///
    /// The slot selection uses round-robin to distribute load across slots.
    /// Each permit acquisition corresponds to one available inference slot.
    async fn acquire(&self) -> (tokio::sync::SemaphorePermit<'_>, tokio::sync::MutexGuard<'_, ModelRequest>) {
        let permit = self
            .semaphore
            .acquire()
            .await
            .expect("semaphore closed");
        // After acquiring, the number of available permits tells us how many
        // concurrent requests are still possible. Use this to compute the slot index
        // in a round-robin fashion. This avoids always using slot 0 and distributes
        // load across all available slots.
        let active = self.slots.len() - self.semaphore.available_permits() - 1;
        let idx = active % self.slots.len();
        let guard = self.slots[idx].lock().await;
        (permit, guard)
    }

    /// Acquire a slot (blocking, for use in spawn_blocking). Returns a
    /// permit and the mutex guard.
    fn acquire_blocking(&self) -> (tokio::sync::SemaphorePermit<'_>, tokio::sync::MutexGuard<'_, ModelRequest>) {
        let permit = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.semaphore.acquire())
        })
        .expect("semaphore closed");
        // Same round-robin logic as acquire()
        let active = self.slots.len() - self.semaphore.available_permits() - 1;
        let idx = active % self.slots.len();
        let guard = self.slots[idx].blocking_lock();
        (permit, guard)
    }
}

/// Engine type selector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineType {
    /// Pure C++ inference (no Python dependency)
    #[default]
    PureCpp,
}

impl EngineType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpp" | "c++" | "native" | "pure_cpp" => Some(EngineType::PureCpp),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
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
    /// Model hidden dimension (used for embeddings)
    pub hidden_size: Option<usize>,
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
            hidden_size: None,
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

/// Parse hidden_size from config.json
fn parse_hidden_size(model_path: &std::path::Path) -> usize {
    let config_path = model_path.join("config.json");
    if !config_path.exists() {
        return 4096; // Default fallback
    }

    std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|content| {
            // Parse config.json as JSON to extract hidden_size
            serde_json::from_str::<serde_json::Value>(&content)
                .ok()
                .and_then(|v| {
                    // Handle nested configs (text_config, model_config)
                    let config = v.get("text_config")
                        .or_else(|| v.get("model_config"))
                        .unwrap_or(&v);

                    config.get("hidden_size").and_then(|h| h.as_u64()).map(|u| u as usize)
                })
        })
        .unwrap_or(4096) // Default fallback
}

/// TurboMind pure C++ engine
pub struct TurboMindCEngine {
    pub(super) model_path: String,
    pub model_name: String,
    state: ModelState,
    pub loaded_at: Option<i64>,
    pub is_ready: std::sync::atomic::AtomicBool,
    pub engine_type: EngineType,

    // C API components
    tm: Option<Arc<TurboMind>>,
    /// Pool of ModelRequest instances for concurrent inference (includes semaphore)
    request_pool: Option<Arc<RequestPool>>,

    // Tokenizer for encoding/decoding
    tokenizer: Option<LMTokenizer>,

    // Configuration
    session_len: i32,
    max_batch_size: i32,
    quant_policy: i32,

    /// Model hidden dimension (from config.json), used for embeddings
    hidden_size: usize,
}

/// Extract logprobs from the C++ output tensors.
///
/// The C++ engine produces three output tensors when `output_logprobs > 0`:
/// - `logprob_indexes`: [batch, seq_len, k] int32 — top-k token indexes per position
/// - `logprob_vals`: [batch, seq_len, k] float32 — log probabilities per index
/// - `logprob_nums`: [batch, seq_len] int32 — how many entries are valid per position
///
/// This function reads these tensors, selects the first (selected token) logprob
/// and the top-k logprobs, and returns them paired with the decoded token text.
///
/// `output_ids` — the generated token IDs (int32 slice)
/// `request` — the completed ModelRequest to read output tensors from
/// `tokenizer` — for decoding token IDs to text
/// `top_k_requested` — number of top logprobs requested (from GenerationParams)
fn extract_logprobs(
    output_ids: &[i32],
    request: &ModelRequest,
    tokenizer: &LMTokenizer,
    top_k_requested: u32,
) -> Option<Vec<TokenLogprob>> {
    // Read logprob_nums to know how many valid entries per position
    let logprob_nums = match request.get_output("logprob_nums") {
        Ok((ptr, size)) => unsafe {
            Some(std::slice::from_raw_parts(ptr as *const i32, size / 4))
        },
        Err(_) => None,
    };
    let logprob_nums = logprob_nums?;

    // Read logprob_vals: [batch=1, seq_len, k]
    let logprob_vals = match request.get_output("logprob_vals") {
        Ok((ptr, size)) => unsafe {
            Some(std::slice::from_raw_parts(ptr as *const f32, size / 4))
        },
        Err(_) => None,
    };
    let logprob_vals = logprob_vals?;

    // Read logprob_indexes: [batch=1, seq_len, k]
    let logprob_indexes = match request.get_output("logprob_indexes") {
        Ok((ptr, size)) => unsafe {
            Some(std::slice::from_raw_parts(ptr as *const i32, size / 4))
        },
        Err(_) => None,
    };
    let logprob_indexes = logprob_indexes?;

    let k = top_k_requested as usize;
    let seq_len = output_ids.len();
    if seq_len == 0 || logprob_nums.len() < seq_len {
        return None;
    }

    // The tensors are flat: [batch * seq_len * k], batch=1
    // index into vals/indexes: row = seq_idx * k, col = logprob_idx
    let mut result = Vec::with_capacity(seq_len);

    for (i, &token_id) in output_ids.iter().enumerate() {
        let num_valid = if i < logprob_nums.len() { logprob_nums[i] as usize } else { 0 };
        if num_valid == 0 {
            // No logprobs available for this position — emit a placeholder
            result.push(TokenLogprob {
                token: tokenizer
                    .decode(&[token_id as u32], true)
                    .ok()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_default(),
                logprob: 0.0,
                bytes: Vec::new(),
                top_logprobs: Vec::new(),
            });
            continue;
        }

        let base_idx = i * k;
        let actual_k = num_valid.min(k);

        // Find the selected token's logprob (where index matches token_id)
        let mut selected_logprob = 0.0f64;
        let mut top_logprobs = Vec::with_capacity(actual_k);
        for j in 0..actual_k {
            let idx = base_idx + j;
            let tok_id = logprob_indexes[idx];
            let lp = logprob_vals[idx] as f64;
            let token_text = tokenizer
                .decode(&[tok_id as u32], true)
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_default();
            let bytes_vec = token_text.as_bytes().to_vec();

            top_logprobs.push(TopLogprob {
                token: token_text.clone(),
                logprob: lp,
                bytes: bytes_vec.clone(),
            });

            if tok_id == token_id {
                selected_logprob = lp;
            }
        }

        // If we didn't find the exact token in top-k, use the first logprob
        // as a fallback (shouldn't happen in normal cases)
        if actual_k == 0 && base_idx < logprob_vals.len() {
            selected_logprob = logprob_vals[base_idx] as f64;
        }

        let token_text = tokenizer
            .decode(&[token_id as u32], true)
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_default();

        result.push(TokenLogprob {
            token: token_text.clone(),
            logprob: selected_logprob,
            bytes: token_text.as_bytes().to_vec(),
            top_logprobs,
        });
    }

    Some(result)
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

        // Parse hidden_size from config.json (needed for embeddings)
        let hidden_size = parse_hidden_size(&model_path_obj);
        if is_awq {
            tracing::info!("Detected AWQ quantized model, enabling quant_policy=4");
        }
        tracing::info!(hidden_size, "Model config parsed");

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

        // Set nnodes=1 to disable distributed mode (avoid LMDEPLOY_DIST_INIT_ADDR requirement)
        engine_config.set_nnodes(1);
        engine_config.set_node_rank(0);

        // Set tensor parallelism sizes (required for TurboMind initialization)
        // These must satisfy: mlp_tp_size == attn_dp_size * attn_tp_size * attn_cp_size
        engine_config.set_attn_tp_size(1);
        engine_config.set_attn_cp_size(1);
        engine_config.set_attn_dp_size(1);
        engine_config.set_mlp_tp_size(1);

        // Set prefill iteration limits
        engine_config.set_num_tokens_per_iter(0);
        engine_config.set_max_prefill_iters(1);

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

        // Create inference request pool for concurrent access (includes semaphore)
        let request_pool = Arc::new(RequestPool::new(&tm, DEFAULT_CONCURRENCY)?);

        let model_name = model_path_obj
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("default")
            .to_string();

        tracing::info!(
            concurrency = DEFAULT_CONCURRENCY,
            "TurboMind C++ engine initialized successfully"
        );

        Ok(Self {
            model_path: model_path.to_string(),
            model_name,
            state: ModelState::Ready,
            loaded_at: Some(unix_timestamp()),
            is_ready: std::sync::atomic::AtomicBool::new(true),
            engine_type: EngineType::PureCpp,
            tm: Some(Arc::new(tm)),
            request_pool: Some(request_pool),
            tokenizer,
            session_len: 65536,
            max_batch_size: 32,
            quant_policy,
            hidden_size,
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
            hidden_size: Some(self.hidden_size),
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
        self.request_pool = None;

        // Re-initialize
        let new_engine = Self::new(new_model_path).await?;
        self.tm = new_engine.tm;
        self.request_pool = new_engine.request_pool;
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
    pub async fn generate(&self, prompt: &str, params: GenerationParams) -> String {
        let (text, _, _) = self.generate_with_metrics(prompt, params).await;
        text
    }

    /// Generate text with TurboMind C++ engine, returning (text, num_tokens, elapsed_ms)
    pub async fn generate_with_metrics(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> (String, usize, f64) {
        let pool = self.request_pool.as_ref().expect("Request pool not initialized");

        // Tokenize input (outside the lock to minimize critical section)
        let input_ids = match &self.tokenizer {
            Some(tokenizer) => match tokenizer.encode(prompt, false, false) {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::error!(error = %e, "Tokenization failed");
                    return (String::new(), 0, 0.0);
                }
            },
            None => {
                tracing::error!("Tokenizer not available");
                return (String::new(), 0, 0.0);
            }
        };

        tracing::debug!(input_len = input_ids.len(), "Tokenized prompt");

        let start = Instant::now();

        // Acquire a slot (semaphore permit + mutex guard for the slot).
        // tokio::sync::Mutex allows the runtime to yield while waiting,
        // enabling true parallel inference without blocking threads.
        let (_permit, mut request) = pool.acquire().await;

        // Prepare input tensors
        let mut input_tensors = TensorMap::new().unwrap();
        let input_ids_shape = [input_ids.len() as i64];
        input_tensors.set_int64(
            "input_ids",
            &input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            &input_ids_shape,
        );
        input_tensors.set_int32("sequence_length", &[input_ids.len() as i32], &[1]);

        // Prepare generation config with HTTP parameters
        let mut gen_cfg = GenConfig::new().unwrap();
        gen_cfg.set_max_new_tokens(params.max_tokens.unwrap_or(512) as i32);
        gen_cfg.set_temperature(params.temperature.unwrap_or(0.7));
        gen_cfg.set_top_p(params.top_p.unwrap_or(0.95));
        gen_cfg.set_top_k(params.top_k.unwrap_or(50) as i32);
        // Apply any additional parameters (min_p, repetition_penalty, seed)
        params.apply_to_gen_config(&mut gen_cfg);

        // Prepare session parameters (use unique ID for each request)
        let session = TM_SessionParam {
            id: unix_timestamp() as u64,
            step: 0,
            start_flag: true,
            end_flag: true,
        };

        // Prepare output tensors
        let mut output_tensors = TensorMap::new().unwrap();

        // Run inference. The request is Send+Sync and the C++ engine handles
        // its own internal synchronization, so we can proceed without holding
        // the tokio mutex during the blocking FFI call.
        match request.forward(
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
                match request.get_output("output_ids") {
                    Ok((data_ptr, size)) => {
                        // data_ptr points to int32 array
                        let num_tokens = size / 4;
                        let output_ids: Vec<i32> = unsafe {
                            std::slice::from_raw_parts(data_ptr as *const i32, num_tokens).to_vec()
                        };

                        // Decode output tokens
                        let text = if let Some(tokenizer) = &self.tokenizer {
                            match tokenizer.decode(
                                &output_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                                true,
                            ) {
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

    /// Generate text with logprobs, returning (text, num_tokens, elapsed_ms, logprobs)
    ///
    /// When `params.logprobs` is true or `params.top_logprobs` is set, this method
    /// extracts log probability information from the C++ engine output tensors.
    pub async fn generate_with_logprobs(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> (String, usize, f64, Option<Vec<TokenLogprob>>) {
        let pool = self.request_pool.as_ref().expect("Request pool not initialized");
        let need_logprobs = params.logprobs.unwrap_or(false) || params.top_logprobs.unwrap_or(0) > 0;
        let top_logprobs_req = params.top_logprobs.unwrap_or(1).max(1);

        // Tokenize input
        let input_ids = match &self.tokenizer {
            Some(tokenizer) => match tokenizer.encode(prompt, false, false) {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::error!(error = %e, "Tokenization failed");
                    return (String::new(), 0, 0.0, None);
                }
            },
            None => {
                tracing::error!("Tokenizer not available");
                return (String::new(), 0, 0.0, None);
            }
        };

        tracing::debug!(input_len = input_ids.len(), "Tokenized prompt");

        let start = Instant::now();

        let (_permit, mut request) = pool.acquire().await;

        // Prepare input tensors
        let mut input_tensors = TensorMap::new().unwrap();
        let input_ids_shape = [input_ids.len() as i64];
        input_tensors.set_int64(
            "input_ids",
            &input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            &input_ids_shape,
        );
        input_tensors.set_int32("sequence_length", &[input_ids.len() as i32], &[1]);

        // Prepare generation config
        let mut gen_cfg = GenConfig::new().unwrap();
        gen_cfg.set_max_new_tokens(params.max_tokens.unwrap_or(512) as i32);
        gen_cfg.set_temperature(params.temperature.unwrap_or(0.7));
        gen_cfg.set_top_p(params.top_p.unwrap_or(0.95));
        gen_cfg.set_top_k(params.top_k.unwrap_or(50) as i32);
        params.apply_to_gen_config(&mut gen_cfg);

        let session = TM_SessionParam {
            id: unix_timestamp() as u64,
            step: 0,
            start_flag: true,
            end_flag: true,
        };

        let mut output_tensors = TensorMap::new().unwrap();

        match request.forward(
            &mut input_tensors,
            &session,
            &gen_cfg,
            false,
            true,
            &mut output_tensors,
        ) {
            Ok(_) => {
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                match request.get_output("output_ids") {
                    Ok((data_ptr, size)) => {
                        let num_tokens = size / 4;
                        let output_ids: Vec<i32> = unsafe {
                            std::slice::from_raw_parts(data_ptr as *const i32, num_tokens).to_vec()
                        };

                        let text = if let Some(tokenizer) = &self.tokenizer {
                            match tokenizer.decode(
                                &output_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                                true,
                            ) {
                                Ok(t) => t,
                                Err(e) => {
                                    tracing::error!(error = %e, "Decoding failed");
                                    format!("[decode error: {}]", e)
                                }
                            }
                        } else {
                            format!("{:?}", output_ids)
                        };

                        let logprobs = if need_logprobs {
                            self.tokenizer
                                .as_ref()
                                .and_then(|tok| {
                                    extract_logprobs(&output_ids, &request, tok, top_logprobs_req)
                                })
                        } else {
                            None
                        };

                        (text, num_tokens, elapsed_ms, logprobs)
                    }
                    Err(e) => {
                        tracing::error!(error = ?e, "Failed to get output_ids");
                        (String::new(), 0, elapsed_ms, None)
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = ?e, "C++ inference failed");
                (String::new(), 0, 0.0, None)
            }
        }
    }

    /// Generate text with streaming output (token-by-token)
    ///
    /// Uses event-driven callbacks from the C++ engine instead of polling.
    /// Each generated token fires a callback that decodes and sends it through the channel.
    pub async fn generate_stream(&self, prompt: &str, params: GenerationParams) -> std::pin::Pin<Box<dyn futures::Stream<Item = String> + Send>> {
        // Tokenize input
        let (input_ids, tokenizer) = match &self.tokenizer {
            Some(t) => {
                let ids = match t.encode(prompt, false, false) {
                    Ok(ids) => ids,
                    Err(e) => {
                        tracing::error!(error = %e, "Tokenization failed");
                        return Box::pin(futures::stream::empty());
                    }
                };
                (ids, t.clone())
            }
            None => {
                tracing::error!("Tokenizer not available");
                return Box::pin(futures::stream::empty());
            }
        };

        let input_ids_vec: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
        let prompt_len = input_ids.len() as i32;

        let pool = self.request_pool.as_ref().expect("Request pool not initialized").clone();
        let params_for_blocking = params.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);

        tokio::task::spawn_blocking(move || {
            let (_permit, mut request) = pool.acquire_blocking();

            // Create callback context
            let ctx = Arc::new(StreamContext { tokenizer, tx });
            let ctx_ptr = Arc::into_raw(ctx) as *mut c_void;

            // Set the token callback before submitting the request
            if let Err(e) = unsafe { request.set_token_callback(token_callback, ctx_ptr) } {
                tracing::error!(error = ?e, "Failed to set token callback");
                let _ = unsafe { Arc::from_raw(ctx_ptr as *const StreamContext) };
                return;
            }

            // Prepare input tensors
            let mut input_tensors = match crate::turbomind_c::TensorMap::new() {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to create tensor map");
                    let _ = unsafe { Arc::from_raw(ctx_ptr as *const StreamContext) };
                    return;
                }
            };
            let input_ids_shape = [input_ids_vec.len() as i64];
            input_tensors.set_int64("input_ids", &input_ids_vec, &input_ids_shape);
            input_tensors.set_int32("sequence_length", &[prompt_len], &[1]);

            // Prepare generation config with HTTP parameters
            let mut gen_cfg = match crate::turbomind_c::GenConfig::new() {
                Ok(g) => g,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to create gen config");
                    let _ = unsafe { Arc::from_raw(ctx_ptr as *const StreamContext) };
                    return;
                }
            };
            gen_cfg.set_max_new_tokens(params_for_blocking.max_tokens.unwrap_or(1024) as i32);
            gen_cfg.set_temperature(params_for_blocking.temperature.unwrap_or(0.7));
            gen_cfg.set_top_p(params_for_blocking.top_p.unwrap_or(0.95));
            gen_cfg.set_top_k(params_for_blocking.top_k.unwrap_or(50) as i32);
            // Apply any additional parameters (min_p, repetition_penalty, seed)
            params_for_blocking.apply_to_gen_config(&mut gen_cfg);

            // Session parameters (unique session ID)
            let session = crate::turbomind_c::TM_SessionParam {
                id: unix_timestamp() as u64,
                step: 0,
                start_flag: true,
                end_flag: true,
            };

            // Submit async forward with stream_output=true
            if let Err(e) = request.forward_async(
                &mut input_tensors,
                &session,
                &gen_cfg,
                true,  // stream_output
                false, // enable_metrics
            ) {
                tracing::error!(error = ?e, "ForwardAsync failed");
                let _ = unsafe { Arc::from_raw(ctx_ptr as *const StreamContext) };
                return;
            }

            // Wait for completion (no polling needed for tokens, only for status)
            loop {
                std::thread::sleep(std::time::Duration::from_millis(5));

                let (status, _seq_len) = match request.get_streaming_state() {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                match status {
                    crate::turbomind_c::TM_RequestStatus::TM_STATUS_FINISH => break,
                    crate::turbomind_c::TM_RequestStatus::TM_STATUS_CANCEL => break,
                    crate::turbomind_c::TM_RequestStatus::TM_STATUS_FAIL => break,
                    crate::turbomind_c::TM_RequestStatus::TM_STATUS_TOO_LONG => break,
                    crate::turbomind_c::TM_RequestStatus::TM_STATUS_INCONSISTENCY => break,
                    _ => continue,
                }
            }

            // Reclaim the Arc to prevent memory leak
            let _ctx = unsafe { Arc::from_raw(ctx_ptr as *const StreamContext) };
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> Option<&LMTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Generate embeddings for text by running token embedding lookup + forward pass.
    ///
    /// Uses `output_last_hidden_state=2` (kGeneration = last token only) to extract
    /// the final hidden state, which represents the semantic embedding of the input.
    ///
    /// Note: This performs a minimal forward pass to get hidden states only. The model
    /// still needs to be fully initialized. If `dimensions` is specified and smaller
    /// than the model's hidden size, returns the first `dimensions` dimensions.
    pub async fn embed(&self, text: &str, dimensions: Option<usize>) -> Vec<f32> {
        let _tm = match &self.tm {
            Some(t) => t,
            None => {
                tracing::error!("TurboMind not initialized");
                return Vec::new();
            }
        };

        // Tokenize input
        let input_ids = match &self.tokenizer {
            Some(tokenizer) => match tokenizer.encode(text, false, false) {
                Ok(ids) => ids,
                Err(e) => {
                    tracing::error!(error = %e, "Tokenization failed for embed");
                    return Vec::new();
                }
            },
            None => {
                tracing::error!("Tokenizer not available for embed");
                return Vec::new();
            }
        };

        if input_ids.is_empty() {
            tracing::warn!("Empty input for embed");
            return Vec::new();
        }

        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
        let batch_size = input_ids_i64.len();

        // Get hidden size from model config (stored during init)
        let hidden_size = self.hidden_size;

        // Default to full hidden size, truncate if dimensions requested

        let target_dims = dimensions.unwrap_or(hidden_size).min(hidden_size);

        tracing::debug!(
            batch_size,
            hidden_size,
            target_dims,
            "embed: prepared input tensors"
        );

        // Acquire a ModelRequest from the pool (must clone Arc for spawn_blocking 'static)
        let pool = match &self.request_pool {
            Some(p) => Arc::clone(p),
            None => {
                tracing::error!("Request pool not initialized");
                return Vec::new();
            }
        };

        // Use blocking task for FFI calls
        let embedding_result = tokio::task::spawn_blocking(move || {
            // Acquire a slot (blocking semaphore + mutex)
            let (_permit, mut request) = pool.acquire_blocking();

            // Prepare input tensors
            let mut input_tensors = match TensorMap::new() {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to create tensor map");
                    return Vec::new();
                }
            };
            let shape = [batch_size as i64];
            input_tensors.set_int64("input_ids", &input_ids_i64, &shape);
            input_tensors.set_int32("sequence_length", &[batch_size as i32], &[1]);

            // Prepare generation config with output_last_hidden_state=2 (kGeneration = last token)
            let mut gen_cfg = match GenConfig::new() {
                Ok(g) => g,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to create gen config");
                    return Vec::new();
                }
            };
            gen_cfg.set_max_new_tokens(1);
            gen_cfg.set_temperature(0.0);
            gen_cfg.set_output_last_hidden_state(2); // kGeneration = last token only

            // Session parameters
            let session = TM_SessionParam {
                id: unix_timestamp() as u64,
                step: 0,
                start_flag: true,
                end_flag: true,
            };

            // Prepare output tensors
            let mut output_tensors = match TensorMap::new() {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to create output tensor map");
                    return Vec::new();
                }
            };

            // Run inference to get last_hidden_state
            if let Err(e) = request.forward(
                &mut input_tensors,
                &session,
                &gen_cfg,
                false, // stream_output
                false, // enable_metrics
                &mut output_tensors,
            ) {
                tracing::error!(error = ?e, "embed forward failed");
                return Vec::new();
            }

            // Extract last_hidden_state from output
            let (data_ptr, size) = match request.get_output("last_hidden_state") {
                Ok((ptr, sz)) => (ptr, sz),
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to get last_hidden_state output");
                    return Vec::new();
                }
            };

            let elem_count = size / 4; // float32 = 4 bytes
            if elem_count == 0 {
                tracing::warn!("Empty last_hidden_state output");
                return Vec::new();
            }

            tracing::debug!(elem_count, "last_hidden_state raw data");

            // The hidden state is [batch_size, hidden_dim] but we only asked for kGeneration
            // (last token), so we should get [1, hidden_dim]
            // However, if output_last_hidden_state=2 returns last prompt token (not generated token),
            // we need to extract correctly. Let's check the actual shape.
            //
            // If we have [1, hidden_dim], extract first `target_dims` elements
            // If we have [batch_size, hidden_dim], extract the LAST row's first `target_dims` elements

            let embedding = if elem_count == hidden_size {
                // Single vector: [hidden_dim]
                let slice = unsafe { std::slice::from_raw_parts(data_ptr as *const f32, elem_count) };
                slice[..target_dims].to_vec()
            } else if elem_count > hidden_size {
                // Multiple vectors: [N, hidden_dim] - take the LAST one (last token)
                let num_vectors = elem_count / hidden_size;
                let start_idx = (num_vectors - 1) * hidden_size;
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        data_ptr.add(start_idx) as *const f32,
                        hidden_size,
                    )
                };
                slice[..target_dims].to_vec()
            } else {
                tracing::warn!(
                    elem_count,
                    hidden_size,
                    "Unexpected last_hidden_state size"
                );
                Vec::new()
            };

            tracing::debug!(embedding_len = embedding.len(), "embed: returning embedding");
            embedding
        })
        .await
        .unwrap_or_default();

        embedding_result
    }

    /// Get schedule metrics
    pub fn get_metrics(&self) -> Result<ScheduleMetrics> {
        let tm = self.tm.as_ref().expect("TurboMind not initialized");
        tm.get_schedule_metrics(0).map_err(|e| {
            AppError::InferenceFailed(format!("Failed to get metrics: {:?}", e))
        })
    }
}

impl std::fmt::Debug for TurboMindCEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurboMindCEngine")
            .field("model_path", &self.model_path)
            .field("model_name", &self.model_name)
            .field("state", &self.state)
            .field("loaded_at", &self.loaded_at)
            .field("engine_type", &self.engine_type)
            .field("quant_policy", &self.quant_policy)
            .field("hidden_size", &self.hidden_size)
            .finish()
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
        assert_eq!(EngineType::from_str("cpp"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("c++"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("native"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("invalid"), None);
    }

    #[test]
    fn test_engine_type_as_str() {
        assert_eq!(EngineType::PureCpp.as_str(), "pure_cpp");
    }

    #[test]
    fn test_detect_awq_quantization() {
        // Test with non-existent path
        let path = std::path::PathBuf::from("/nonexistent/path");
        assert!(!detect_awq_quantization(&path));
    }
}
