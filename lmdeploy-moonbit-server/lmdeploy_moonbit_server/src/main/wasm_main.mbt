// src/main/wasm_main.mbt
// Wasm main entry point for LMDeploy MoonBit API Server
// This is the _start function that WasmEdge will call when loading the WASM module

// Target: wasm32-unknown-unknown with WASI support
// Runtime: WasmEdge 0.13+
// Memory Limit: 10 MB

mod wasi
mod wasi::entry

/// Wasm module entry point
/// This function is called by WasmEdge when the WASM module is loaded
pub extern "C" fn _start() -> Int {
  // Initialize the Wasm application
  let state = wasi::entry::WasmAppState::init()

  if !state.is_initialized {
    // Return error code 1 if initialization failed
    return 1
  }

  // Start the HTTP server
  let server_result = wasi::entry::start_http_server(state)

  match server_result {
    Ok(_) => 0,
    Err(msg) => {
      // Log error and return error code
      wasi_log_error("Failed to start server: " + msg)
      1
    }
  }
}

/// Alternative entry point with WasmEdge initialization
/// This provides more control over the WasmEdge runtime configuration
pub extern "C" fn _initialize() -> Int {
  // Configure WasmEdge runtime
  let config = wasi::WasmEdgeConfig::default()

  // Initialize runtime
  let runtime_result = wasi::WasmEdgeRuntime::new(config)

  match runtime_result {
    Ok(runtime) => {
      let init_result = runtime.init()

      match init_result {
        Ok(_) => 0,
        Err(msg) => {
          wasi_log_error("Runtime init failed: " + msg)
          1
        }
      }
    }
    Err(msg) => {
      wasi_log_error("Runtime creation failed: " + msg)
      1
    }
  }
}

/// WASI logging function for errors
/// This writes to stderr via WASI fd_write
fn wasi_log_error(msg: String) {
  // TODO: Implement using WASI fd_write to stderr (fd=2)
  // For now, this is a placeholder
}

/// Memory export for Wasm monitoring
/// Returns the current memory usage in bytes
pub extern "C" fn lmdeploy_memory_usage() -> Int {
  // TODO: Implement memory tracking
  // This should return the actual WASM memory usage
  10 * 1024 * 1024  // 10 MB placeholder
}

/// Version export for runtime detection
pub extern "C" fn lmdeploy_version() -> Int {
  0x00010000  // v0.1.0
}

/// Feature flags export
/// Returns a bitmask of enabled features
pub extern "C" fn lmdeploy_features() -> Int {
  // Feature flags:
  // 0x01: HTTP API
  // 0x02: gRPC API
  // 0x04: SSE Streaming
  // 0x08: Tokenize Cache
  // 0x10: Rate Limiting
  // 0x20: Model Management
  // 0x40: Prometheus Metrics
  0x7F  // All features enabled
}
