// src/config/quantization.mbt
// Quantization configuration for AWQ, GPTQ, and other quantization methods
// Reference: lmdeploy/turbomind/models/quantization.h
// Reference: HuggingFace quantization_config format

use username/lmdeploy_moonbit_server.model::weights

// ============================================================================
// AWQ Quantization Configuration
// ============================================================================

/// AWQ (Activation-aware Weight Quantization) configuration
/// AWQ 4-bit quantization uses per-channel scales and zeros
pub struct AwqConfig {
  /// Number of bits for quantization (typically 4)
  bits: Int
  /// Group size for quantization (-1 means per-channel)
  group_size: Int
  /// Quantization method (e.g., "awq")
  quant_method: String
  /// Version string
  version: String
  /// Whether to use symmetric quantization
  symmetric: Bool
  /// True if zeros point is set (usually true for AWQ)
  zero_point: Bool
  /// Whether to pack weights (4-bit packing)
  pack: Bool
  /// Weight data type after quantization
  weight_dtype: weights.WeightDtype
  /// Module replacement configuration (layers to quantize)
  quant_modules: Array[String]
  /// Skip list (modules to skip quantization)
  skip_modules: Array[String]
  /// Whether this is an AWQ quantized model
  is_enabled: Bool
}

pub fn AwqConfig::default() -> AwqConfig {
  {
    bits: 4,
    group_size: 128,
    quant_method: "awq",
    version: "gemm",
    symmetric: true,
    zero_point: true,
    pack: true,
    weight_dtype: weights.WeightDtype::Float16,
    quant_modules: [],
    skip_modules: [],
    is_enabled: false,
  }
}

pub fn AwqConfig::new(bits: Int, group_size: Int) -> AwqConfig {
  {
    bits,
    group_size,
    quant_method: "awq",
    version: "gemm",
    symmetric: true,
    zero_point: true,
    pack: true,
    weight_dtype: weights.WeightDtype::Float16,
    quant_modules: [],
    skip_modules: [],
    is_enabled: true,
  }
}

pub fn AwqConfig::bits(self: AwqConfig) -> Int {
  self.bits
}

pub fn AwqConfig::group_size(self: AwqConfig) -> Int {
  self.group_size
}

pub fn AwqConfig::quant_method(self: AwqConfig) -> String {
  self.quant_method
}

pub fn AwqConfig::version(self: AwqConfig) -> String {
  self.version
}

pub fn AwqConfig::is_symmetric(self: AwqConfig) -> Bool {
  self.symmetric
}

pub fn AwqConfig::has_zero_point(self: AwqConfig) -> Bool {
  self.zero_point
}

pub fn AwqConfig::is_packed(self: AwqConfig) -> Bool {
  self.pack
}

pub fn AwqConfig::weight_dtype(self: AwqConfig) -> weights.WeightDtype {
  self.weight_dtype
}

pub fn AwqConfig::is_enabled(self: AwqConfig) -> Bool {
  self.is_enabled
}

pub fn AwqConfig::to_debug_string(self: AwqConfig) -> String {
  "AwqConfig {"
    ++ " bits:" ++ int_to_string(self.bits)
    ++ ", group_size:" ++ int_to_string(self.group_size)
    ++ ", quant_method:\"" ++ self.quant_method ++ "\""
    ++ ", version:\"" ++ self.version ++ "\""
    ++ ", symmetric:" ++ bool_to_string(self.symmetric)
    ++ ", zero_point:" ++ bool_to_string(self.zero_point)
    ++ ", pack:" ++ bool_to_string(self.pack)
    ++ ", is_enabled:" ++ bool_to_string(self.is_enabled)
    ++ " }"
}

// ============================================================================
// GPTQ Quantization Configuration
// ============================================================================

/// GPTQ (Gradient-based Post-Training Quantization) configuration
pub struct GptqConfig {
  /// Number of bits for quantization (typically 4)
  bits: Int
  /// Group size for quantization
  group_size: Int
  /// Quantization method (e.g., "gptq")
  quant_method: String
  /// Symmetric quantization
  symmetric: Bool
  /// True if zeros point is set
  zero_point: Bool
  /// Description string
  description: String
  /// Block index for weight unpacking
  block_index: Array[Int]
  /// Quantization scale
  scale: Float
  /// Whether this is a GPTQ quantized model
  is_enabled: Bool
}

pub fn GptqConfig::default() -> GptqConfig {
  {
    bits: 4,
    group_size: 128,
    quant_method: "gptq",
    symmetric: true,
    zero_point: true,
    description: "GPTQ quantized",
    block_index: [],
    scale: 1.0,
    is_enabled: false,
  }
}

pub fn GptqConfig::is_enabled(self: GptqConfig) -> Bool {
  self.is_enabled
}

pub fn GptqConfig::bits(self: GptqConfig) -> Int {
  self.bits
}

pub fn GptqConfig::group_size(self: GptqConfig) -> Int {
  self.group_size
}

// ============================================================================
// General Quantization Configuration
// ============================================================================

/// Unified quantization configuration
pub enum QuantizationConfig {
  /// No quantization
  None
  /// AWQ quantization
  Awq(AwqConfig)
  /// GPTQ quantization
  Gptq(GptqConfig)
  /// Other quantization methods (bitsandbytes, etc.)
  Other(String)
}

pub fn QuantizationConfig::default() -> QuantizationConfig {
  None
}

pub fn QuantizationConfig::is_enabled(self: QuantizationConfig) -> Bool {
  match self {
    None => false
    Awq(cfg) => cfg.is_enabled()
    Gptq(cfg) => cfg.is_enabled()
    Other(_) => true
  }
}

pub fn QuantizationConfig::quant_method(self: QuantizationConfig) -> String {
  match self {
    None => "none"
    Awq(cfg) => cfg.quant_method()
    Gptq(cfg) => cfg.quant_method()
    Other(method) => method
  }
}

pub fn QuantizationConfig::bits(self: QuantizationConfig) -> Int {
  match self {
    None => 16
    Awq(cfg) => cfg.bits()
    Gptq(cfg) => cfg.bits()
    Other(_) => 4
  }
}

pub fn QuantizationConfig::group_size(self: QuantizationConfig) -> Int {
  match self {
    None => -1
    Awq(cfg) => cfg.group_size()
    Gptq(cfg) => cfg.group_size()
    Other(_) => 128
  }
}

pub fn QuantizationConfig::to_debug_string(self: QuantizationConfig) -> String {
  match self {
    None => "QuantizationConfig { None }"
    Awq(cfg) => "QuantizationConfig { " ++ cfg.to_debug_string() ++ " }"
    Gptq(cfg) => "QuantizationConfig { GptqConfig { bits:" ++ int_to_string(cfg.bits()) ++ " } }"
    Other(method) => "QuantizationConfig { Other:\"" ++ method ++ "\" }"
  }
}

// ============================================================================
// Parsing Functions
// ============================================================================

/// Parse quantization_config from config.json
/// Returns QuantizationConfig based on the quantization method detected
pub fn parse_quantization_config(json: String) -> QuantizationConfig {
  if json == "" {
    return QuantizationConfig::default()
  }

  // Find quantization_config block
  let quant_start = json.find("\"quantization_config\":")
  if quant_start < 0 {
    return QuantizationConfig::default()
  }

  // Extract the quantization_config object
  let obj_start = json.find_from("{", quant_start)
  if obj_start < 0 {
    return QuantizationConfig::default()
  }

  let obj_end = find_matching_brace(json, obj_start)
  if obj_end < 0 {
    return QuantizationConfig::default()
  }

  let quant_json = json.substring(obj_start, obj_end + 1)

  // Detect quantization method
  let quant_method = parse_json_string_field(quant_json, "quant_method", "")

  if quant_method == "awq" {
    parse_awq_config(quant_json)
  } else if quant_method == "gptq" {
    parse_gptq_config(quant_json)
  } else {
    // Unknown quantization method
    QuantizationConfig::Other(quant_method)
  }
}

/// Parse AWQ configuration from quantization_config JSON
fn parse_awq_config(json: String) -> QuantizationConfig {
  let bits = parse_json_int_field(json, "bits", 4)
  let group_size = parse_json_int_field(json, "group_size", 128)
  let version = parse_json_string_field(json, "version", "gemm")
  let symmetric = parse_json_bool_field(json, "symmetric", true)
  let zero_point = parse_json_bool_field(json, "zero_point", true)
  let pack = parse_json_bool_field(json, "pack", true)

  // Parse weight_dtype if present
  let weight_dtype_str = parse_json_string_field(json, "weight_dtype", "float16")
  let weight_dtype = weights.WeightDtype::from_string(weight_dtype_str)

  let cfg = AwqConfig {
    bits,
    group_size,
    quant_method: "awq",
    version,
    symmetric,
    zero_point,
    pack,
    weight_dtype,
    quant_modules: [],
    skip_modules: [],
    is_enabled: true,
  }

  QuantizationConfig::Awq(cfg)
}

/// Parse GPTQ configuration from quantization_config JSON
fn parse_gptq_config(json: String) -> QuantizationConfig {
  let bits = parse_json_int_field(json, "bits", 4)
  let group_size = parse_json_int_field(json, "group_size", 128)
  let symmetric = parse_json_bool_field(json, "symmetric", true)
  let description = parse_json_string_field(json, "description", "GPTQ quantized")

  let cfg = GptqConfig {
    bits,
    group_size,
    quant_method: "gptq",
    symmetric,
    zero_point: true,
    description,
    block_index: [],
    scale: 1.0,
    is_enabled: true,
  }

  QuantizationConfig::Gptq(cfg)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Find matching closing brace for an opening brace
fn find_matching_brace(json: String, open_pos: Int) -> Int {
  let mut depth = 1
  let mut idx = open_pos + 1

  while idx < json.length() && depth > 0 {
    let c = json[idx]
    if c == '{' {
      depth = depth + 1
    } else if c == '}' {
      depth = depth - 1
    }
    idx = idx + 1
  }

  if depth == 0 {
    idx - 1
  } else {
    -1
  }
}

fn parse_json_string_field(json: String, field: String, default: String) -> String {
  let pattern = "\"" ++ field ++ "\":\""
  let idx = json.find(pattern)
  if idx >= 0 {
    let start = idx + pattern.length()
    let end_idx = json.find_from("\"", start)
    if end_idx >= 0 {
      json.substring(start, end_idx)
    } else {
      default
    }
  } else {
    default
  }
}

fn parse_json_int_field(json: String, field: String, default: Int) -> Int {
  let pattern = "\"" ++ field ++ "\":"
  let idx = json.find(pattern)
  if idx >= 0 {
    let start = idx + pattern.length()
    let mut start = start
    while start < json.length() && (json[start] == ' ' || json[start] == '\t' || json[start] == '\n') {
      start = start + 1
    }
    let end_idx = json.find_from(",", start)
    let end_idx2 = json.find_from("}", start)
    let end = if end_idx >= 0 && (end_idx2 < 0 || end_idx < end_idx2) { end_idx } else { end_idx2 }
    if end >= 0 {
      let num_str = json.substring(start, end)
      string_to_int(num_str)
    } else {
      default
    }
  } else {
    default
  }
}

fn parse_json_bool_field(json: String, field: String, default: Bool) -> Bool {
  let pattern = "\"" ++ field ++ "\":"
  let idx = json.find(pattern)
  if idx >= 0 {
    let start = idx + pattern.length()
    let mut start = start
    while start < json.length() && (json[start] == ' ' || json[start] == '\t' || json[start] == '\n') {
      start = start + 1
    }
    let end_idx = json.find_from(",", start)
    let end_idx2 = json.find_from("}", start)
    let end = if end_idx >= 0 && (end_idx2 < 0 || end_idx < end_idx2) { end_idx } else { end_idx2 }
    if end >= 0 {
      let value_str = json.substring(start, end)
      if value_str == "true" {
        true
      } else if value_str == "false" {
        false
      } else {
        default
      }
    } else {
      default
    }
  } else {
    default
  }
}

fn string_to_int(s: String) -> Int {
  let mut result = 0
  let mut idx = 0
  while idx < s.length() {
    let c = s[idx]
    if c >= '0' && c <= '9' {
      result = result * 10 + (c - '0')
    }
    idx = idx + 1
  }
  result
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
