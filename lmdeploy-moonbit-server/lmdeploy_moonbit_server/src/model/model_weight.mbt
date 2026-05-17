// src/model/model_weight.mbt
// Complete ModelWeight hierarchy for TurboMind model weights
// Reference: src/turbomind/models/model_weight.h, decoder_layer_weight.h,
//            attention_weight.h, ffn_weight.h, moe_weight.h, delta_net_weight.h

use username/lmdeploy_moonbit_server/error
use username/lmdeploy_moonbit_server.model::weights

// ============================================================================
// Configuration Types
// ============================================================================

/// Rope (Rotary Position Embedding) configuration
pub struct RopeConfig {
  /// Rope type (0=none, 1=linear, 2=dynamic, etc.)
  type_: Int
  /// Base for rotary position computation
  base: Float
  /// Rotary dimension
  dim: Int
  /// Scaling factor for positions
  factor: Float
  /// Maximum position embeddings
  max_position_embeddings: Int
  /// Yarn attention factor
  yarn_attention_factor: Float
  /// Yarn beta fast
  yarn_beta_fast: Float
  /// Yarn beta slow
  yarn_beta_slow: Float
  /// Llama3 low frequency factor
  llama3_low_freq_factor: Float
  /// Llama3 high frequency factor
  llama3_high_freq_factor: Float
  /// Llama3 original max position embeddings
  llama3_original_max_position_embeddings: Int
  /// Multi-rope section for multimodal models [dim1, dim2, dim3]
  mrope_section: (Int, Int, Int)
}

pub fn RopeConfig::default() -> RopeConfig {
  {
    type_: 0,
    base: 10000.0,
    dim: 0,
    factor: 1.0,
    max_position_embeddings: 0,
    yarn_attention_factor: 1.0,
    yarn_beta_fast: 32.0,
    yarn_beta_slow: 1.0,
    llama3_low_freq_factor: 1.0,
    llama3_high_freq_factor: 4.0,
    llama3_original_max_position_embeddings: 0,
    mrope_section: (0, 0, 0),
  }
}

/// Attention layer configuration
pub struct AttentionConfig {
  /// Hidden dimension
  hidden_dim: Int
  /// Head dimension
  head_dim: Int
  /// Number of attention heads
  head_num: Int
  /// Number of key-value heads (for GQA/MQA)
  kv_head_num: Int
  /// KV LoRA rank (for Multi-Head Latent Attention)
  kv_lora_rank: Int
  /// Q LoRA rank
  q_lora_rank: Int
  /// QK RoPE dimension
  qk_rope_dim: Int
  /// Value head dimension
  v_head_dim: Int
  /// Tensor parallel size
  tp_size: Int
  /// Tensor parallel rank
  tp_rank: Int
  /// Data type
  data_type: weights.WeightDtype
  /// Sliding window size (0 = no window)
  window_size: Int
  /// Whether to use output gate
  output_gate: Bool
  /// Rope configuration
  rope: RopeConfig
  /// QK no-PE dimension (for partial RoPE)
  qk_nope_dim: Int
  /// Softmax scale (0 = compute from head_dim)
  softmax_scale: Float
  /// Whether to use LogN attention scaling
  use_logn_attn: Bool
}

pub fn AttentionConfig::default() -> AttentionConfig {
  {
    hidden_dim: 0,
    head_dim: 128,
    head_num: 0,
    kv_head_num: 0,
    kv_lora_rank: 0,
    q_lora_rank: 0,
    qk_rope_dim: 0,
    v_head_dim: 0,
    tp_size: 1,
    tp_rank: 0,
    data_type: weights.WeightDtype::Float16,
    window_size: 0,
    output_gate: false,
    rope: RopeConfig::default(),
    qk_nope_dim: 0,
    softmax_scale: 0.0,
    use_logn_attn: false,
  }
}

/// Feed-forward network configuration
pub struct FfnConfig {
  /// Hidden dimension
  hidden_dim: Int
  /// Intermediate size (FFN hidden dimension)
  inter_size: Int
  /// Activation type (0=gelu, 1=relu, 2=silu, etc.)
  act_type: Int
  /// Whether to fuse SiLU activation
  fuse_silu: Bool
  /// Whether this is an expert FFN (for MoE)
  is_expert: Bool
  /// Data type
  data_type: weights.WeightDtype
  /// Tensor parallel size
  tp_size: Int
  /// Tensor parallel rank
  tp_rank: Int
}

pub fn FfnConfig::default() -> FfnConfig {
  {
    hidden_dim: 0,
    inter_size: 0,
    act_type: 1, // SiLU
    fuse_silu: false,
    is_expert: false,
    data_type: weights.WeightDtype::Float16,
    tp_size: 1,
    tp_rank: 0,
  }
}

/// MoE (Mixture of Experts) configuration
pub struct MoeConfig {
  /// Number of experts
  expert_num: Int
  /// Number of experts per token
  experts_per_token: Int
  /// Activation type
  act_type: Int
  /// Whether to fuse SiLU activation
  fuse_silu: Bool
  /// Whether to normalize top-k probabilities
  norm_topk_prob: Bool
  /// Top-k method ("greedy" or "group_limited_greedy")
  topk_method: String
  /// Scoring function ("softmax" or "sigmoid")
  scoring_func: String
  /// Top-k group
  topk_group: Int
  /// Number of groups
  n_group: Int
  /// Router number of groups
  router_n_groups: Int
  /// Routed scale factor
  routed_scale: Float
  /// Data type
  data_type: weights.WeightDtype
}

pub fn MoeConfig::default() -> MoeConfig {
  {
    expert_num: 0,
    experts_per_token: 0,
    act_type: 1, // SiLU
    fuse_silu: false,
    norm_topk_prob: true,
    topk_method: "greedy",
    scoring_func: "softmax",
    topk_group: 0,
    n_group: 1,
    router_n_groups: 1,
    routed_scale: 1.0,
    data_type: weights.WeightDtype::Float16,
  }
}

/// DeltaNet (linear attention) configuration
pub struct DeltaNetConfig {
  /// Hidden dimension
  hidden_dim: Int
  /// Number of key heads
  num_k_heads: Int
  /// Number of value heads
  num_v_heads: Int
  /// Key head dimension
  key_head_dim: Int
  /// Value head dimension
  value_head_dim: Int
  /// 1D convolution kernel size
  d_conv: Int
  /// Data type
  data_type: weights.WeightDtype
  /// Tensor parallel size
  tp_size: Int
  /// Tensor parallel rank
  tp_rank: Int
}

pub fn DeltaNetConfig::default() -> DeltaNetConfig {
  {
    hidden_dim: 0,
    num_k_heads: 0,
    num_v_heads: 0,
    key_head_dim: 0,
    value_head_dim: 0,
    d_conv: 4,
    data_type: weights.WeightDtype::Float16,
    tp_size: 1,
    tp_rank: 0,
  }
}

/// Normalization layer configuration
pub struct NormConfig {
  /// Dimension
  dim: Int
  /// Data type
  data_type: weights.WeightDtype
  /// Epsilon for numerical stability
  norm_eps: Float
}

pub fn NormConfig::default() -> NormConfig {
  {
    dim: 0,
    data_type: weights.WeightDtype::Float16,
    norm_eps: 0.00001,
  }
}

/// Linear layer configuration
pub struct LinearConfig {
  /// Input dimension
  input_dim: Int
  /// Output dimension
  output_dim: Int
  /// Data type
  data_type: weights.WeightDtype
  /// Whether this layer has bias
  has_bias: Bool
}

pub fn LinearConfig::default() -> LinearConfig {
  {
    input_dim: 0,
    output_dim: 0,
    data_type: weights.WeightDtype::Float16,
    has_bias: false,
  }
}

/// Model weight configuration
pub struct ModelWeightConfig {
  /// Tensor parallel size
  tp_size: Int
  /// Tensor parallel rank
  tp_rank: Int
  /// Data type
  data_type: weights.WeightDtype
  /// Hidden units (hidden_size)
  hidden_units: Int
}

pub fn ModelWeightConfig::default() -> ModelWeightConfig {
  {
    tp_size: 1,
    tp_rank: 0,
    data_type: weights.WeightDtype::Float16,
    hidden_units: 0,
  }
}

pub fn ModelWeightConfig::from_model_config(
  num_layers: Int,
  num_heads: Int,
  num_kv_heads: Int,
  hidden_size: Int,
  head_dim: Int,
  intermediate_size: Int,
  data_type_str: String,
  tp_size: Int,
  tp_rank: Int,
) -> ModelWeightConfig {
  let dtype = weights.WeightDtype::from_string(data_type_str)
  {
    tp_size,
    tp_rank,
    data_type: dtype,
    hidden_units: hidden_size,
  }
}

// ============================================================================
// Weight Types
// ============================================================================

/// Normalization layer weights (RMSNorm/LayerNorm)
pub struct NormWeight {
  /// Weight tensor (gamma for RMSNorm, scale for LayerNorm)
  weight: Option[weights.WeightTensor]
  /// Configuration
  config: NormConfig
  /// Whether weights are loaded
  is_loaded: Bool
}

pub fn NormWeight::new(config: NormConfig) -> NormWeight {
  {
    weight: None,
    config,
    is_loaded: false,
  }
}

pub fn NormWeight::with_weight(self: NormWeight, weight: weights.WeightTensor) -> NormWeight {
  {
    weight: Some(weight),
    is_loaded: true,
    ..self
  }
}

pub fn NormWeight::dim(self: NormWeight) -> Int {
  self.config.dim
}

pub fn NormWeight::is_loaded(self: NormWeight) -> Bool {
  self.is_loaded
}

/// Linear layer weights (dense/linear transformation)
pub struct LinearWeight {
  /// Main weight tensor
  weight: Option[weights.WeightTensor]
  /// Bias tensor (optional)
  bias: Option[weights.WeightTensor]
  /// Scales for quantization (optional)
  scales: Option[weights.WeightTensor]
  /// Zeros for quantization (optional)
  zeros: Option[weights.WeightTensor]
  /// Configuration
  config: LinearConfig
  /// Whether weights are loaded
  is_loaded: Bool
}

pub fn LinearWeight::new(config: LinearConfig) -> LinearWeight {
  {
    weight: None,
    bias: None,
    scales: None,
    zeros: None,
    config,
    is_loaded: false,
  }
}

pub fn LinearWeight::with_weight(self: LinearWeight, weight: weights.WeightTensor) -> LinearWeight {
  {
    weight: Some(weight),
    is_loaded: true,
    ..self
  }
}

pub fn LinearWeight::with_bias(self: LinearWeight, bias: weights.WeightTensor) -> LinearWeight {
  { bias: Some(bias), ..self }
}

pub fn LinearWeight::input_dim(self: LinearWeight) -> Int {
  self.config.input_dim
}

pub fn LinearWeight::output_dim(self: LinearWeight) -> Int {
  self.config.output_dim
}

pub fn LinearWeight::is_loaded(self: LinearWeight) -> Bool {
  self.is_loaded
}

/// Attention layer weights
pub struct AttentionWeight {
  /// QKV fused projection (for non-MLA attention)
  w_qkv: Option[LinearWeight]
  /// Q projection (for MLA or separate QKV)
  q_proj: Option[LinearWeight]
  /// Q_a projection (for MLA)
  q_a_proj: Option[LinearWeight]
  /// Q_b projection (for MLA)
  q_b_proj: Option[LinearWeight]
  /// KV_a projection (for MLA)
  kv_a_proj: Option[LinearWeight]
  /// Output projection
  wo: LinearWeight
  /// Q normalization (for MLA)
  q_norm: Option<NormWeight>
  /// K normalization (for MLA)
  k_norm: Option<NormWeight>
  /// Q_a layer norm (for MLA)
  q_a_layernorm: Option<NormWeight>
  /// KV_a layer norm (for MLA)
  kv_a_layernorm: Option<NormWeight>
  /// Sink tokens (for sink attention)
  sinks: Option[weights.WeightTensor]
  /// Configuration
  config: AttentionConfig
  /// Whether weights are loaded
  is_loaded: Bool
}

pub fn AttentionWeight::new(config: AttentionConfig) -> AttentionWeight {
  let linear_cfg = LinearConfig {
    input_dim: config.hidden_dim,
    output_dim: config.hidden_dim,
    data_type: config.data_type,
    has_bias: false,
  }
  {
    w_qkv: None,
    q_proj: None,
    q_a_proj: None,
    q_b_proj: None,
    kv_a_proj: None,
    wo: LinearWeight::new(linear_cfg),
    q_norm: None,
    k_norm: None,
    q_a_layernorm: None,
    kv_a_layernorm: None,
    sinks: None,
    config,
    is_loaded: false,
  }
}

pub fn AttentionWeight::with_w_qkv(self: AttentionWeight, w_qkv: LinearWeight) -> AttentionWeight {
  { w_qkv: Some(w_qkv), is_loaded: true, ..self }
}

pub fn AttentionWeight::with_q_proj(self: AttentionWeight, q_proj: LinearWeight) -> AttentionWeight {
  { q_proj: Some(q_proj), is_loaded: true, ..self }
}

pub fn AttentionWeight::with_wo(self: AttentionWeight, wo: LinearWeight) -> AttentionWeight {
  { wo, is_loaded: true, ..self }
}

pub fn AttentionWeight::is_mla(self: AttentionWeight) -> Bool {
  self.config.kv_lora_rank > 0
}

pub fn AttentionWeight::hidden_dim(self: AttentionWeight) -> Int {
  self.config.hidden_dim
}

pub fn AttentionWeight::head_dim(self: AttentionWeight) -> Int {
  self.config.head_dim
}

pub fn AttentionWeight::head_num(self: AttentionWeight) -> Int {
  self.config.head_num
}

pub fn AttentionWeight::kv_head_num(self: AttentionWeight) -> Int {
  self.config.kv_head_num
}

pub fn AttentionWeight::is_loaded(self: AttentionWeight) -> Bool {
  self.is_loaded
}

/// Feed-forward network weights
pub struct FfnWeight {
  /// First linear transformation (gate)
  w1: Option<LinearWeight>
  /// Third linear transformation (up)
  w3: Option<LinearWeight>
  /// Second linear transformation (down)
  w2: Option<LinearWeight>
  /// Fused w1+w3 (for optimized inference)
  w1w3: Option<LinearWeight>
  /// Configuration
  config: FfnConfig
  /// Whether weights are loaded
  is_loaded: Bool
}

pub fn FfnWeight::new(config: FfnConfig) -> FfnWeight {
  {
    w1: None,
    w3: None,
    w2: None,
    w1w3: None,
    config,
    is_loaded: false,
  }
}

pub fn FfnWeight::with_w1(self: FfnWeight, w1: LinearWeight) -> FfnWeight {
  { w1: Some(w1), is_loaded: true, ..self }
}

pub fn FfnWeight::with_w3(self: FfnWeight, w3: LinearWeight) -> FfnWeight {
  { w3: Some(w3), is_loaded: true, ..self }
}

pub fn FfnWeight::with_w2(self: FfnWeight, w2: LinearWeight) -> FfnWeight {
  { w2: Some(w2), is_loaded: true, ..self }
}

pub fn FfnWeight::with_w1w3(self: FfnWeight, w1w3: LinearWeight) -> FfnWeight {
  { w1w3: Some(w1w3), is_loaded: true, ..self }
}

pub fn FfnWeight::hidden_dim(self: FfnWeight) -> Int {
  self.config.hidden_dim
}

pub fn FfnWeight::inter_size(self: FfnWeight) -> Int {
  self.config.inter_size
}

pub fn FfnWeight::is_loaded(self: FfnWeight) -> Bool {
  self.is_loaded
}

/// Mixture of Experts (MoE) weights
pub struct MoeWeight {
  /// Gate router weights
  gate: Option<LinearWeight>
  /// Shared gate weights (optional)
  shared_gate: Option[LinearWeight]
  /// Array of expert FFN weights
  experts: Array[FfnWeight]
  /// Score correction bias (for normalized routing)
  score_correction_bias: Option[weights.WeightTensor]
  /// Configuration
  config: MoeConfig
  /// Whether weights are loaded
  is_loaded: Bool
}

pub fn MoeWeight::new(config: MoeConfig) -> MoeWeight {
  let expert_num = config.expert_num
  let mut experts = []
  for _ in 0..expert_num {
    let ffn_cfg = FfnConfig {
      is_expert: true,
      ..config
    }
    experts = experts.concat([FfnWeight::new(ffn_cfg)])
  }
  {
    gate: None,
    shared_gate: None,
    experts,
    score_correction_bias: None,
    config,
    is_loaded: false,
  }
}

pub fn MoeWeight::with_gate(self: MoeWeight, gate: LinearWeight) -> MoeWeight {
  { gate: Some(gate), is_loaded: true, ..self }
}

pub fn MoeWeight::with_shared_gate(self: MoeWeight, shared_gate: LinearWeight) -> MoeWeight {
  { shared_gate: Some(shared_gate), is_loaded: true, ..self }
}

pub fn MoeWeight::num_experts(self: MoeWeight) -> Int {
  self.config.expert_num
}

pub fn MoeWeight::expert(self: MoeWeight, i: Int) -> Option[FfnWeight] {
  if i >= 0 && i < self.experts.length() {
    Some(self.experts[i])
  } else {
    None
  }
}

pub fn MoeWeight::is_loaded(self: MoeWeight) -> Bool {
  self.is_loaded
}

/// DeltaNet (linear attention) weights
pub struct DeltaNetWeight {
  /// Input projection (all heads)
  in_proj_all: Option[LinearWeight]
  /// Output projection
  out_proj: Option<LinearWeight>
  /// Normalization
  norm: Option<NormWeight>
  /// 1D convolution weights
  conv1d: Option[weights.WeightTensor]
  /// Log A parameter
  a_log: Option[weights.WeightTensor]
  /// DT bias
  dt_bias: Option[weights.WeightTensor]
  /// Configuration
  config: DeltaNetConfig
  /// Whether weights are loaded
  is_loaded: Bool
}

pub fn DeltaNetWeight::new(config: DeltaNetConfig) -> DeltaNetWeight {
  {
    in_proj_all: None,
    out_proj: None,
    norm: None,
    conv1d: None,
    a_log: None,
    dt_bias: None,
    config,
    is_loaded: false,
  }
}

pub fn DeltaNetWeight::with_in_proj_all(self: DeltaNetWeight, in_proj: LinearWeight) -> DeltaNetWeight {
  { in_proj_all: Some(in_proj), is_loaded: true, ..self }
}

pub fn DeltaNetWeight::with_out_proj(self: DeltaNetWeight, out_proj: LinearWeight) -> DeltaNetWeight {
  { out_proj: Some(out_proj), is_loaded: true, ..self }
}

pub fn DeltaNetWeight::with_norm(self: DeltaNetWeight, norm: NormWeight) -> DeltaNetWeight {
  { norm: Some(norm), is_loaded: true, ..self }
}

pub fn DeltaNetWeight::hidden_dim(self: DeltaNetWeight) -> Int {
  self.config.hidden_dim
}

pub fn DeltaNetWeight::is_loaded(self: DeltaNetWeight) -> Bool {
  self.is_loaded
}

/// Decoder layer weights (single transformer layer)
pub struct DecoderLayerWeight {
  /// Self-attention weights
  attention: Option<AttentionWeight>
  /// Linear attention (DeltaNet) weights
  linear_attn: Option[DeltaNetWeight]
  /// Feed-forward network weights
  feed_forward: Option[FfnWeight]
  /// MoE FFN weights (if using MoE)
  moe_ffn: Option[MoeWeight]
  /// Pre-attention normalization
  attention_norm: Option<NormWeight]
  /// Pre-FFN normalization
  ffn_norm: Option<NormWeight>
  /// Layer index (0 to num_layers-1)
  layer_idx: Int
  /// Whether weights are loaded
  is_loaded: Bool
}

pub fn DecoderLayerWeight::new(layer_idx: Int) -> DecoderLayerWeight {
  {
    attention: None,
    linear_attn: None,
    feed_forward: None,
    moe_ffn: None,
    attention_norm: None,
    ffn_norm: None,
    layer_idx,
    is_loaded: false,
  }
}

pub fn DecoderLayerWeight::with_attention(self: DecoderLayerWeight, attention: AttentionWeight) -> DecoderLayerWeight {
  { attention: Some(attention), is_loaded: true, ..self }
}

pub fn DecoderLayerWeight::with_linear_attn(self: DecoderLayerWeight, linear_attn: DeltaNetWeight) -> DecoderLayerWeight {
  { linear_attn: Some(linear_attn), is_loaded: true, ..self }
}

pub fn DecoderLayerWeight::with_feed_forward(self: DecoderLayerWeight, feed_forward: FfnWeight) -> DecoderLayerWeight {
  { feed_forward: Some(feed_forward), is_loaded: true, ..self }
}

pub fn DecoderLayerWeight::with_moe_ffn(self: DecoderLayerWeight, moe_ffn: MoeWeight) -> DecoderLayerWeight {
  { moe_ffn: Some(moe_ffn), is_loaded: true, ..self }
}

pub fn DecoderLayerWeight::with_attention_norm(self: DecoderLayerWeight, norm: NormWeight) -> DecoderLayerWeight {
  { attention_norm: Some(norm), is_loaded: true, ..self }
}

pub fn DecoderLayerWeight::with_ffn_norm(self: DecoderLayerWeight, norm: NormWeight) -> DecoderLayerWeight {
  { ffn_norm: Some(norm), is_loaded: true, ..self }
}

pub fn DecoderLayerWeight::layer_idx(self: DecoderLayerWeight) -> Int {
  self.layer_idx
}

pub fn DecoderLayerWeight::is_loaded(self: DecoderLayerWeight) -> Bool {
  self.is_loaded
}

/// Model weights (root weight module)
pub struct ModelWeight {
  /// Token embedding weights
  tok_embeddings: Option[weights.WeightTensor]
  /// Output projection (lm_head)
  output: Option[LinearWeight]
  /// Final normalization
  norm: Option<NormWeight>
  /// Decoder layers (40 layers for Qwen3.6-35B)
  layers: Array[DecoderLayerWeight]
  /// Configuration
  config: ModelWeightConfig
  /// Derived fields (computed in prepare())
  data_type: weights.WeightDtype
  hidden_units: Int
  vocab_size: Int
  vocab_size_padded: Int
  embedding_size: Int
  num_layer: Int
  head_dim: Int
  kv_head_num: Int
  /// Whether weights are loaded
  is_loaded: Bool
}

pub fn ModelWeight::new(config: ModelWeightConfig) -> ModelWeight {
  let num_layers = 40 // Qwen3.6-35B has 40 layers
  let mut layers = []
  for i in 0..num_layers {
    layers = layers.concat([DecoderLayerWeight::new(i)])
  }
  {
    tok_embeddings: None,
    output: None,
    norm: None,
    layers,
    config,
    data_type: config.data_type,
    hidden_units: config.hidden_units,
    vocab_size: 0,
    vocab_size_padded: 0,
    embedding_size: 0,
    num_layer: num_layers,
    head_dim: 128,
    kv_head_num: 0,
    is_loaded: false,
  }
}

pub fn ModelWeight::with_tok_embeddings(self: ModelWeight, tok_emb: weights.WeightTensor) -> ModelWeight {
  { tok_embeddings: Some(tok_emb), is_loaded: true, ..self }
}

pub fn ModelWeight::with_output(self: ModelWeight, output: LinearWeight) -> ModelWeight {
  { output: Some(output), is_loaded: true, ..self }
}

pub fn ModelWeight::with_norm(self: ModelWeight, norm: NormWeight) -> ModelWeight {
  { norm: Some(norm), is_loaded: true, ..self }
}

pub fn ModelWeight::with_layers(self: ModelWeight, layers: Array[DecoderLayerWeight]) -> ModelWeight {
  { layers, is_loaded: true, ..self }
}

pub fn ModelWeight::layer(self: ModelWeight, i: Int) -> Option[DecoderLayerWeight] {
  if i >= 0 && i < self.layers.length() {
    Some(self.layers[i])
  } else {
    None
  }
}

pub fn ModelWeight::num_layers(self: ModelWeight) -> Int {
  self.num_layer
}

pub fn ModelWeight::is_loaded(self: ModelWeight) -> Bool {
  self.is_loaded
}

/// Prepare model weights after loading
/// Computes derived fields and validates weights
pub fn ModelWeight::prepare(self: ModelWeight) -> Result[ModelWeight, error.LmDeployError] {
  // Validate that at least the token embeddings are loaded
  match self.tok_embeddings {
    None => Err(error::internal_error("tok_embeddings not loaded"))
    Some(_) => Ok(self)
  }
}

/// Verify model weights are complete
/// Returns list of missing weight names
pub fn ModelWeight::verify(self: ModelWeight) -> Result[ModelWeight, Array[String]] {
  let mut missing = []

  // Check token embeddings
  match self.tok_embeddings {
    None => missing = missing.concat(["tok_embeddings"])
    Some(_) => ()
  }

  // Check output
  match self.output {
    None => missing = missing.concat(["output"])
    Some(_) => ()
  }

  // Check norm
  match self.norm {
    None => missing = missing.concat(["norm"])
    Some(_) => ()
  }

  // Check layers
  for i in 0..self.layers.length() {
    let layer = self.layers[i]
    if !layer.is_loaded() {
      missing = missing.concat(["layer_" ++ int_to_string(i)])
    }
  }

  if missing.length() > 0 {
    Err(missing)
  } else {
    Ok(self)
  }
}

// ============================================================================
// Builder functions for constructing weights from config
// ============================================================================

/// Create ModelWeightConfig from model parameters
pub fn create_model_weight_config(
  hidden_size: Int,
  head_dim: Int,
  intermediate_size: Int,
  num_layers: Int,
  num_heads: Int,
  num_kv_heads: Int,
  vocab_size: Int,
  data_type: String,
  tp_size: Int,
  tp_rank: Int,
) -> ModelWeightConfig {
  let dtype = weights.WeightDtype::from_string(data_type)
  {
    tp_size,
    tp_rank,
    data_type: dtype,
    hidden_units: hidden_size,
  }
}

/// Create AttentionConfig from model parameters
pub fn create_attention_config(
  hidden_size: Int,
  head_dim: Int,
  num_heads: Int,
  num_kv_heads: Int,
  data_type: weights.WeightDtype,
  window_size: Int,
  rope_dim: Int,
  tp_size: Int,
  tp_rank: Int,
) -> AttentionConfig {
  {
    hidden_dim: hidden_size,
    head_dim,
    head_num: num_heads,
    kv_head_num: num_kv_heads,
    kv_lora_rank: 0,
    q_lora_rank: 0,
    qk_rope_dim: 0,
    v_head_dim: head_dim,
    tp_size,
    tp_rank,
    data_type,
    window_size,
    output_gate: false,
    rope: RopeConfig {
      dim: rope_dim,
      ..RopeConfig::default()
    },
    qk_nope_dim: 0,
    softmax_scale: 0.0,
    use_logn_attn: false,
  }
}

/// Create FfnConfig from model parameters
pub fn create_ffn_config(
  hidden_size: Int,
  intermediate_size: Int,
  data_type: weights.WeightDtype,
  tp_size: Int,
  tp_rank: Int,
) -> FfnConfig {
  {
    hidden_dim: hidden_size,
    inter_size: intermediate_size,
    act_type: 1, // SiLU
    fuse_silu: false,
    is_expert: false,
    data_type,
    tp_size,
    tp_rank,
  }
}

/// Create MoeConfig from model parameters
pub fn create_moe_config(
  expert_num: Int,
  experts_per_token: Int,
  hidden_size: Int,
  intermediate_size: Int,
  data_type: weights.WeightDtype,
) -> MoeConfig {
  {
    expert_num,
    experts_per_token,
    act_type: 1, // SiLU
    fuse_silu: false,
    norm_topk_prob: true,
    topk_method: "greedy",
    scoring_func: "softmax",
    topk_group: 0,
    n_group: 1,
    router_n_groups: 1,
    routed_scale: 1.0,
    data_type,
  }
}

/// Create DeltaNetConfig from model parameters
pub fn create_deltanet_config(
  hidden_size: Int,
  num_k_heads: Int,
  num_v_heads: Int,
  key_head_dim: Int,
  value_head_dim: Int,
  data_type: weights.WeightDtype,
  tp_size: Int,
  tp_rank: Int,
) -> DeltaNetConfig {
  {
    hidden_dim: hidden_size,
    num_k_heads,
    num_v_heads,
    key_head_dim,
    value_head_dim,
    d_conv: 4,
    data_type,
    tp_size,
    tp_rank,
  }
}

/// Create NormConfig from model parameters
pub fn create_norm_config(dim: Int, data_type: weights.WeightDtype, eps: Float) -> NormConfig {
  { dim, data_type, norm_eps: eps }
}

/// Create LinearConfig from model parameters
pub fn create_linear_config(
  input_dim: Int,
  output_dim: Int,
  data_type: weights.WeightDtype,
  has_bias: Bool,
) -> LinearConfig {
  { input_dim, output_dim, data_type, has_bias }
}

// ============================================================================
// Helper functions
// ============================================================================

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
