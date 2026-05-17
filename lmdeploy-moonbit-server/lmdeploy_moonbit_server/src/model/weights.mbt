// src/model/weights.mbt
// SafeTensors weight loading and management for GPU memory
// Reference: lmdeploy/turbomind/models/model_weight.h, safetensors spec
//
// SafeTensors format:
// - Last 8 bytes: offset to JSON header (from end of file)
// - JSON header contains tensor metadata (dtype, shape, data_offsets)
// - Tensor data follows header in contiguous memory

use username/lmdeploy_moonbit_server/error
use username/lmdeploy_moonbit_server/config

/// Supported data types for weights
pub enum WeightDtype {
  Float32
  Float16
  BFloat16
  Int8
  UInt8
  Int32
  Int64
  Unknown(String)
}

pub fn WeightDtype::to_string(self: WeightDtype) -> String {
  match self {
    Float32 => "F32"
    Float16 => "F16"
    BFloat16 => "BF16"
    Int8 => "I8"
    UInt8 => "U8"
    Int32 => "I32"
    Int64 => "I64"
    Unknown(s) => s
  }
}

pub fn WeightDtype::from_string(s: String) -> WeightDtype {
  if s == "F32" { Float32 }
  else if s == "F16" { Float16 }
  else if s == "BF16" { BFloat16 }
  else if s == "I8" { Int8 }
  else if s == "U8" { UInt8 }
  else if s == "I32" { Int32 }
  else if s == "I64" { Int64 }
  else { Unknown(s) }
}

pub fn WeightDtype::size_bytes(self: WeightDtype) -> Int {
  match self {
    Float32 => 4
    Float16 => 2
    BFloat16 => 2
    Int8 => 1
    UInt8 => 1
    Int32 => 4
    Int64 => 8
    Unknown(_) => 4
  }
}

/// Tensor shape (dimensions)
pub struct TensorShape {
  dims: Array[Int]
}

pub fn TensorShape::new(dims: Array[Int]) -> TensorShape {
  { dims }
}

pub fn TensorShape::from_array(arr: Array[Int]) -> TensorShape {
  TensorShape::new(arr)
}

pub fn TensorShape::rank(self: TensorShape) -> Int {
  self.dims.length()
}

pub fn TensorShape::num_elements(self: TensorShape) -> Int {
  let mut count = 1
  for dim in self.dims {
    count = count * dim
  }
  count
}

pub fn TensorShape::to_string(self: TensorShape) -> String {
  "[" ++ array_to_string(self.dims) ++ "]"
}

/// Data offsets in safetensors file
pub struct DataOffsets {
  begin: Int64
  end: Int64
}

pub fn DataOffsets::new(begin: Int64, end: Int64) -> DataOffsets {
  { begin, end }
}

pub fn DataOffsets::length(self: DataOffsets) -> Int64 {
  self.end - self.begin
}

/// Weight tensor metadata from safetensors
pub struct WeightTensor {
  /// Tensor name (e.g., "model.layers.0.self_attn.q_proj.weight")
  name: String
  /// Data type
  dtype: WeightDtype
  /// Tensor shape
  shape: TensorShape
  /// Data offsets in file
  data_offsets: DataOffsets
  /// Whether tensor is loaded on GPU
  is_loaded: Bool
  /// GPU memory address (placeholder for FFI)
  gpu_address: Int64
}

pub fn WeightTensor::new(name: String, dtype: WeightDtype, shape: TensorShape, data_offsets: DataOffsets) -> WeightTensor {
  {
    name,
    dtype,
    shape,
    data_offsets,
    is_loaded: false,
    gpu_address: 0L,
  }
}

pub fn WeightTensor::name(self: WeightTensor) -> String {
  self.name
}

pub fn WeightTensor::dtype(self: WeightTensor) -> WeightDtype {
  self.dtype
}

pub fn WeightTensor::shape(self: WeightTensor) -> TensorShape {
  self.shape
}

pub fn WeightTensor::num_elements(self: WeightTensor) -> Int {
  self.shape.num_elements()
}

pub fn WeightTensor::byte_size(self: WeightTensor) -> Int64 {
  let element_bytes = self.dtype.size_bytes()
  self.data_offsets.length()
}

pub fn WeightTensor::is_loaded(self: WeightTensor) -> Bool {
  self.is_loaded
}

pub fn WeightTensor::gpu_address(self: WeightTensor) -> Int64 {
  self.gpu_address
}

pub fn WeightTensor::with_loaded(self: WeightTensor, gpu_addr: Int64) -> WeightTensor {
  {
    is_loaded: true,
    gpu_address: gpu_addr,
    ..self
  }
}

pub fn WeightTensor::to_debug_string(self: WeightTensor) -> String {
  "WeightTensor{"
    ++ "name:\"" ++ self.name ++ "\""
    ++ ", dtype:" ++ self.dtype.to_string()
    ++ ", shape:" ++ self.shape.to_string()
    ++ ", bytes:" ++ int64_to_string(self.byte_size())
    ++ ", loaded:" ++ bool_to_string(self.is_loaded)
    ++ "}"
}

/// SafeTensors file header metadata
pub struct SafeTensorsHeader {
  /// Number of tensors in file
  num_tensors: Int
  /// All tensor metadata
  tensors: Array[WeightTensor]
  /// Total size of all tensor data
  total_data_size: Int64
  /// Model metadata (optional)
  metadata: Map[String, String]
}

pub fn SafeTensorsHeader::new() -> SafeTensorsHeader {
  {
    num_tensors: 0,
    tensors: [],
    total_data_size: 0L,
    metadata: Map::new(),
  }
}

pub fn SafeTensorsHeader::with_tensors(self: SafeTensorsHeader, tensors: Array[WeightTensor]) -> SafeTensorsHeader {
  let mut total_size = 0L
  for tensor in tensors {
    total_size = total_size + tensor.byte_size()
  }
  {
    num_tensors: tensors.length(),
    tensors,
    total_data_size: total_size,
    metadata: self.metadata,
  }
}

pub fn SafeTensorsHeader::with_metadata(self: SafeTensorsHeader, metadata: Map[String, String]) -> SafeTensorsHeader {
  { metadata, ..self }
}

pub fn SafeTensorsHeader::tensors(self: SafeTensorsHeader) -> Array[WeightTensor] {
  self.tensors
}

pub fn SafeTensorsHeader::num_tensors(self: SafeTensorsHeader) -> Int {
  self.num_tensors
}

pub fn SafeTensorsHeader::total_data_size(self: SafeTensorsHeader) -> Int64 {
  self.total_data_size
}

pub fn SafeTensorsHeader::get_tensor(self: SafeTensorsHeader, name: String) -> Option[WeightTensor] {
  for tensor in self.tensors {
    if tensor.name() == name {
      return Some(tensor)
    }
  }
  None
}

/// SafeTensors file representation
pub struct SafeTensorsFile {
  /// File path
  path: String
  /// Parsed header
  header: SafeTensorsHeader
  /// Whether file is loaded
  is_loaded: Bool
  /// File size in bytes
  file_size: Int64
}

pub fn SafeTensorsFile::new(path: String) -> SafeTensorsFile {
  {
    path,
    header: SafeTensorsHeader::new(),
    is_loaded: false,
    file_size: 0L,
  }
}

pub fn SafeTensorsFile::with_header(self: SafeTensorsFile, header: SafeTensorsHeader, file_size: Int64) -> SafeTensorsFile {
  {
    header,
    is_loaded: true,
    file_size,
    ..self
  }
}

pub fn SafeTensorsFile::path(self: SafeTensorsFile) -> String {
  self.path
}

pub fn SafeTensorsFile::header(self: SafeTensorsFile) -> SafeTensorsHeader {
  self.header
}

pub fn SafeTensorsFile::is_loaded(self: SafeTensorsFile) -> Bool {
  self.is_loaded
}

pub fn SafeTensorsFile::file_size(self: SafeTensorsFile) -> Int64 {
  self.file_size
}

/// Weight container - manages all safetensors files for a model
pub struct WeightContainer {
  /// Model path
  model_path: String
  /// All safetensors files
  files: Array[SafeTensorsFile]
  /// All tensors indexed by name
  tensor_index: Map[String, WeightTensor]
  /// Total GPU memory used
  total_gpu_bytes: Int64
  /// Whether weights are loaded on GPU
  is_gpu_loaded: Bool
}

pub fn WeightContainer::new(model_path: String) -> WeightContainer {
  {
    model_path,
    files: [],
    tensor_index: Map::new(),
    total_gpu_bytes: 0L,
    is_gpu_loaded: false,
  }
}

pub fn WeightContainer::with_files(self: WeightContainer, files: Array[SafeTensorsFile]) -> WeightContainer {
  let mut tensor_index = Map::new()
  let mut total_size = 0L

  for file in files {
    for tensor in file.header().tensors() {
      tensor_index = tensor_index.insert(tensor.name(), tensor)
      total_size = total_size + tensor.byte_size()
    }
  }

  {
    files,
    tensor_index,
    total_gpu_bytes: total_size,
    ..self
  }
}

pub fn WeightContainer::with_gpu_loaded(self: WeightContainer) -> WeightContainer {
  { is_gpu_loaded: true, ..self }
}

pub fn WeightContainer::model_path(self: WeightContainer) -> String {
  self.model_path
}

pub fn WeightContainer::files(self: WeightContainer) -> Array[SafeTensorsFile] {
  self.files
}

pub fn WeightContainer::num_files(self: WeightContainer) -> Int {
  self.files.length()
}

pub fn WeightContainer::num_tensors(self: WeightContainer) -> Int {
  self.tensor_index.length()
}

pub fn WeightContainer::total_gpu_bytes(self: WeightContainer) -> Int64 {
  self.total_gpu_bytes
}

pub fn WeightContainer::is_gpu_loaded(self: WeightContainer) -> Bool {
  self.is_gpu_loaded
}

pub fn WeightContainer::get_tensor(self: WeightContainer, name: String) -> Option[WeightTensor] {
  self.tensor_index.get(name)
}

pub fn WeightContainer::get_tensor_by_pattern(self: WeightContainer, pattern: String) -> Array[WeightTensor] {
  let mut results = []
  for (name, tensor) in self.tensor_index {
    if name.contains(pattern) {
      results = results.concat([tensor])
    }
  }
  results
}

/// Parse safetensors header from JSON content
/// Returns SafeTensorsHeader with tensor metadata
pub fn parse_safetensors_header(json: String) -> Result[SafeTensorsHeader, error.LmDeployError] {
  if json == "" {
    return Err(error::internal_error("empty safetensors header"))
  }

  let mut tensors = []
  let mut metadata = Map::new()

  // Parse "__metadata__" section if present
  let metadata_start = json.find("\"__metadata__\":")
  if metadata_start >= 0 {
    // Extract metadata (simplified - just record that it exists)
    metadata = metadata.insert("has_metadata", "true")
  }

  // Parse tensor entries
  // Format: "tensor_name": {"dtype": "F32", "shape": [1, 128], "data_offsets": [0, 512]}
  let mut idx = 0
  while idx < json.length() {
    // Find next tensor name (quoted string followed by :{)
    let name_start = json.find_from("\"", idx)
    if name_start < 0 { break }
    let name_end = json.find_from("\"", name_start + 1)
    if name_end < 0 { break }

    let tensor_name = json.substring(name_start + 1, name_end)
    if tensor_name == "__metadata__" {
      idx = name_end + 1
      continue
    }

    // Find the opening { after the name
    let obj_start = json.find_from("{", name_end)
    if obj_start < 0 { break }

    // Find matching closing }
    let obj_end = find_matching_brace(json, obj_start)
    if obj_end < 0 { break }

    let obj_content = json.substring(obj_start, obj_end + 1)

    // Parse tensor metadata
    match parse_tensor_metadata(tensor_name, obj_content) {
      Ok(tensor) => tensors = tensors.concat([tensor]),
      Err(_) => () // Skip invalid tensors
    }

    idx = obj_end + 1
  }

  let header = SafeTensorsHeader::new().with_tensors(tensors).with_metadata(metadata)
  Ok(header)
}

/// Parse individual tensor metadata from JSON object
fn parse_tensor_metadata(name: String, json: String) -> Result[WeightTensor, error.LmDeployError] {
  let dtype = WeightDtype::from_string(parse_json_string_field(json, "dtype", "F32"))

  // Parse shape array: [dim1, dim2, ...]
  let shape = parse_shape_array(json)

  // Parse data_offsets: [begin, end]
  let offsets = parse_data_offsets(json)

  Ok(WeightTensor::new(name, dtype, shape, offsets))
}

/// Parse shape array from JSON
fn parse_shape_array(json: String) -> TensorShape {
  let pattern = "\"shape\":["
  let idx = json.find(pattern)
  if idx < 0 {
    return TensorShape::new([])
  }

  let start = idx + pattern.length()
  let end_idx = json.find_from("]", start)
  if end_idx < 0 {
    return TensorShape::new([])
  }

  let array_str = json.substring(start, end_idx)
  parse_int_array(array_str)
}

/// Parse data offsets [begin, end]
fn parse_data_offsets(json: String) -> DataOffsets {
  let pattern = "\"data_offsets\":["
  let idx = json.find(pattern)
  if idx < 0 {
    return DataOffsets::new(0L, 0L)
  }

  let start = idx + pattern.length()
  let end_idx = json.find_from("]", start)
  if end_idx < 0 {
    return DataOffsets::new(0L, 0L)
  }

  let array_str = json.substring(start, end_idx)
  let values = parse_int64_array(array_str)
  if values.length() >= 2 {
    DataOffsets::new(values[0], values[1])
  } else {
    DataOffsets::new(0L, 0L)
  }
}

/// Parse array of integers from JSON array string
fn parse_int_array(s: String) -> Array[Int] {
  let mut result = []
  let mut current = 0
  let mut has_number = false

  for i in 0..s.length() {
    let c = s[i]
    if c >= '0' && c <= '9' {
      current = current * 10 + (c - '0')
      has_number = true
    } else if c == ',' || c == ' ' {
      if has_number {
        result = result.concat([current])
        current = 0
        has_number = false
      }
    }
  }

  if has_number {
    result = result.concat([current])
  }

  result
}

/// Parse array of int64 from JSON array string
fn parse_int64_array(s: String) -> Array[Int64] {
  let mut result = []
  let mut current: Int64 = 0L
  let mut has_number = false

  for i in 0..s.length() {
    let c = s[i]
    if c >= '0' && c <= '9' {
      current = current * 10L + int_to_int64(c - '0')
      has_number = true
    } else if c == ',' || c == ' ' {
      if has_number {
        result = result.concat([current])
        current = 0L
        has_number = false
      }
    }
  }

  if has_number {
    result = result.concat([current])
  }

  result
}

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

/// Load safetensors files from model directory
pub fn load_safetensors_files(model_path: String, cfg: config.model.ModelConfig) -> Result[WeightContainer, error.LmDeployError] {
  // List safetensors files in model directory
  let files = list_safetensors_files(model_path)

  if files.length() == 0 {
    return Err(error::internal_error("no safetensors files found in " ++ model_path))
  }

  let mut loaded_files = []
  for file_path in files {
    match load_safetensors_file(file_path) {
      Ok(file) => loaded_files = loaded_files.concat([file]),
      Err(e) => return Err(e)
    }
  }

  let container = WeightContainer::new(model_path).with_files(loaded_files)
  Ok(container)
}

/// List all safetensors files in model directory
fn list_safetensors_files(model_path: String) -> Array[String] {
  // Stub - requires WASI filesystem FFI
  // Common patterns: model.safetensors, pytorch_model-00001-of-00002.safetensors
  []
}

/// Load a single safetensors file and parse its header
fn load_safetensors_file(path: String) -> Result[SafeTensorsFile, error.LmDeployError] {
  // Read last 8 bytes to get header offset
  // Requires WASI FFI for file I/O

  // Stub: return empty file
  Ok(SafeTensorsFile::new(path).with_header(SafeTensorsHeader::new(), 0L))
}

/// Load weights to GPU memory
pub fn load_weights_to_gpu(container: WeightContainer) -> Result[WeightContainer, error.LmDeployError] {
  // Requires TurboMind FFI to allocate GPU memory and load weights
  // For now, just mark as loaded

  Ok(container.with_gpu_loaded())
}

/// Helper function to parse JSON string field
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

// Helper functions

fn array_to_string(arr: Array[Int]) -> String {
  if arr.length() == 0 {
    return ""
  }

  let mut result = int_to_string(arr[0])
  for i in 1..arr.length() {
    result = result ++ ", " ++ int_to_string(arr[i])
  }
  result
}

fn int64_to_string(i: Int64) -> String {
  // Simplified - for actual implementation needs proper Int64 conversion
  int_to_string(int64_to_int(i))
}

fn int64_to_int(i: Int64) -> Int {
  // Stub conversion
  0
}

fn int_to_int64(i: Int) -> Int64 {
  // Stub conversion
  0L
}

fn bool_to_string(b: Bool) -> String {
  if b { "true" } else { "false" }
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
