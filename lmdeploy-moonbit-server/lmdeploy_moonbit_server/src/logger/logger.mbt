// src/logger/logger.mbt
// Structured logging system for LMDeploy MoonBit server
// Reference: lmdeploy/serve/openai/api_server.py logger usage, Rust logging patterns
//
// Features:
// - Structured JSON logging (when json_format=true)
// - Log levels: DEBUG, INFO, WARN, ERROR
// - Request ID tracking
// - Timestamp and component fields

/// Log level enumeration
pub enum LogLevel {
  Debug
  Info
  Warn
  Error
}

pub fn LogLevel::to_string(self : LogLevel) -> String {
  match self {
    Debug => "debug"
    Info => "info"
    Warn => "warn"
    Error => "error"
  }
}

pub fn LogLevel::from_string(level: String) -> LogLevel {
  match level {
    "debug" => Debug
    "DEBUG" => Debug
    "Debug" => Debug
    "info" => Info
    "INFO" => Info
    "Info" => Info
    "warn" => Warn
    "WARN" => Warn
    "Warn" => Warn
    "warning" => Warn
    "WARNING" => Warn
    "error" => Error
    "ERROR" => Error
    "Error" => Error
    _ => Info
  }
}

pub fn LogLevel::ordinal(self : LogLevel) -> Int {
  match self {
    Debug => 0
    Info => 1
    Warn => 2
    Error => 3
  }
}

pub fn LogLevel::should_log(self : LogLevel, min_level: LogLevel) -> Bool {
  self.ordinal() >= min_level.ordinal()
}

/// Logger configuration
pub struct LoggerConfig {
  level: LogLevel
  json_format: Bool
  component: String
  include_timestamp: Bool
  include_request_id: Bool
}

pub fn LoggerConfig::default() -> LoggerConfig {
  {
    level: Info,
    json_format: true,
    component: "lmdeploy-moonbit-server",
    include_timestamp: true,
    include_request_id: true,
  }
}

pub fn LoggerConfig::new(level: LogLevel, json_format: Bool) -> LoggerConfig {
  let defaults = LoggerConfig::default()
  { level, json_format, component: defaults.component, include_timestamp: defaults.include_timestamp, include_request_id: defaults.include_request_id }
}

pub fn LoggerConfig::with_component(self: LoggerConfig, component: String) -> LoggerConfig {
  { level: self.level, json_format: self.json_format, component, include_timestamp: self.include_timestamp, include_request_id: self.include_request_id }
}

pub fn LoggerConfig::with_level(self: LoggerConfig, level: LogLevel) -> LoggerConfig {
  { level, json_format: self.json_format, component: self.component, include_timestamp: self.include_timestamp, include_request_id: self.include_request_id }
}

pub fn LoggerConfig::with_json_format(self: LoggerConfig, json_format: Bool) -> LoggerConfig {
  { level: self.level, json_format, component: self.component, include_timestamp: self.include_timestamp, include_request_id: self.include_request_id }
}

pub fn LoggerConfig::is_debug(self: LoggerConfig) -> Bool {
  self.level == Debug
}

pub fn LoggerConfig::is_info(self: LoggerConfig) -> Bool {
  self.level == Info || self.level == Debug
}

pub fn LoggerConfig::is_warn_enabled(self: LoggerConfig) -> Bool {
  self.level.ordinal() <= Warn.ordinal()
}

/// Structured log entry
pub struct LogEntry {
  level: LogLevel
  message: String
  component: String
  timestamp: Int64
  request_id: Option[String]
  fields: Map[String, String]
  error: Option[String]
}

pub fn LogEntry::new(level: LogLevel, message: String) -> LogEntry {
  LogEntry {
    level,
    message,
    component: "",
    timestamp: current_timestamp_ms(),
    request_id: None,
    fields: Map([], capacity=0),
    error: None,
  }
}

pub fn LogEntry::with_component(self: LogEntry, component: String) -> LogEntry {
  { level: self.level, message: self.message, component, timestamp: self.timestamp, request_id: self.request_id, fields: self.fields, error: self.error }
}

pub fn LogEntry::with_request_id(self: LogEntry, request_id: String) -> LogEntry {
  { level: self.level, message: self.message, component: self.component, timestamp: self.timestamp, request_id: Some(request_id), fields: self.fields, error: self.error }
}

pub fn LogEntry::with_field(self: LogEntry, key: String, value: String) -> LogEntry {
  let updated_fields = self.fields.insert(key, value)
  { level: self.level, message: self.message, component: self.component, timestamp: self.timestamp, request_id: self.request_id, fields: updated_fields, error: self.error }
}

pub fn LogEntry::with_error(self: LogEntry, error: String) -> LogEntry {
  { level: self.level, message: self.message, component: self.component, timestamp: self.timestamp, request_id: self.request_id, fields: self.fields, error: Some(error) }
}

pub fn LogEntry::with_fields(self: LogEntry, new_fields: Map[String, String]) -> LogEntry {
  let mut updated_fields = self.fields
  for (key, value) in new_fields {
    updated_fields = updated_fields.insert(key, value)
  }
  { level: self.level, message: self.message, component: self.component, timestamp: self.timestamp, request_id: self.request_id, fields: updated_fields, error: self.error }
}

/// Logger instance
pub struct Logger {
  config: LoggerConfig
}

pub fn Logger::new(config: LoggerConfig) -> Logger {
  { config }
}

pub fn Logger::default() -> Logger {
  { config: LoggerConfig::default() }
}

pub fn Logger::with_component(component: String) -> Logger {
  { config: LoggerConfig::default().with_component(component) }
}

pub fn Logger::debug(self: Logger, message: String) -> Unit {
  self.log(Debug, message, None, Map([], capacity=0))
}

pub fn Logger::debug_with_fields(self: Logger, message: String, fields: Map[String, String]) -> Unit {
  self.log(Debug, message, None, fields)
}

pub fn Logger::info(self: Logger, message: String) -> Unit {
  self.log(Info, message, None, Map([], capacity=0))
}

pub fn Logger::info_with_fields(self: Logger, message: String, fields: Map[String, String]) -> Unit {
  self.log(Info, message, None, fields)
}

pub fn Logger::warn(self: Logger, message: String) -> Unit {
  self.log(Warn, message, None, Map([], capacity=0))
}

pub fn Logger::warn_with_error(self: Logger, message: String, error: String) -> Unit {
  self.log(Warn, message, Some(error), Map([], capacity=0))
}

pub fn Logger::warn_with_fields(self: Logger, message: String, fields: Map[String, String]) -> Unit {
  self.log(Warn, message, None, fields)
}

pub fn Logger::error(self: Logger, message: String) -> Unit {
  self.log(Error, message, None, Map([], capacity=0))
}

pub fn Logger::error_with_error(self: Logger, message: String, error: String) -> Unit {
  self.log(Error, message, Some(error), Map([], capacity=0))
}

pub fn Logger::error_with_fields(self: Logger, message: String, error: Option[String], fields: Map[String, String]) -> Unit {
  self.log(Error, message, error, fields)
}

fn Logger::log(self: Logger, level: LogLevel, message: String, error: Option[String], fields: Map[String, String]) -> Unit {
  if !level.should_log(self.config.level) {
    ()
  } else {
    let entry = LogEntry::new(level, message)
      .with_component(self.config.component)
      .with_fields(fields)

    let entry = match error {
      Some(err) => entry.with_error(err),
      None => entry
    }

    let formatted = format_log_entry(entry, self.config.json_format, self.config.include_timestamp)
    output_log(formatted)
  }
}

/// Format log entry as JSON or plain text
fn format_log_entry(entry: LogEntry, json_format: Bool, include_timestamp: Bool) -> String {
  if json_format {
    format_json_log(entry, include_timestamp)
  } else {
    format_plain_log(entry, include_timestamp)
  }
}

/// Format log entry as structured JSON
fn format_json_log(entry: LogEntry, include_timestamp: Bool) -> String {
  let level_str = LogLevel::to_string(entry.level)
  let timestamp_str = if include_timestamp {
    int64_to_string(entry.timestamp)
  } else {
    ""
  }

  // Build JSON string manually (no JSON library in MoonBit stdlib)
  let mut json = "{"

  // Add timestamp if enabled
  if include_timestamp {
    json = json ++ "\"timestamp\":" ++ timestamp_str ++ ","
  }

  // Add level
  json = json ++ "\"level\":\"" ++ level_str ++ "\","

  // Add component
  if entry.component != "" {
    json = json ++ "\"component\":\"" ++ entry.component ++ "\","
  }

  // Add message
  json = json ++ "\"message\":\"" ++ escape_json_string(entry.message) ++ "\","

  // Add request_id if present
  match entry.request_id {
    Some(rid) => json = json ++ "\"request_id\":\"" ++ rid ++ "\",",
    None => ()
  }

  // Add error if present
  match entry.error {
    Some(err) => json = json ++ "\"error\":\"" ++ escape_json_string(err) ++ "\",",
    None => ()
  }

  // Add custom fields
  let field_count = entry.fields.length()
  if field_count > 0 {
    let mut idx = 0
    for (key, value) in entry.fields {
      json = json ++ "\"" ++ key ++ "\":\"" ++ escape_json_string(value) ++ "\""
      idx = idx + 1
      if idx < field_count {
        json = json ++ ","
      }
    }
  } else {
    // Remove trailing comma
    let len = json.length()
    if json[len - 1] == ',' {
      json = json.substring(0, len - 1)
    }
  }

  json = json ++ "}"
  json
}

/// Format log entry as plain text
fn format_plain_log(entry: LogEntry, include_timestamp: Bool) -> String {
  let level_str = LogLevel::to_string(entry.level)
  let mut result = ""

  if include_timestamp {
    result = result ++ "[" ++ int64_to_string(entry.timestamp) ++ "] "
  }

  result = result ++ "[" ++ level_str ++ "]"

  if entry.component != "" {
    result = result ++ " [" ++ entry.component ++ "]"
  }

  result = result ++ " " ++ entry.message

  match entry.request_id {
    Some(rid) => result = result ++ " (req=" ++ rid ++ ")",
    None => ()
  }

  match entry.error {
    Some(err) => result = result ++ " | error: " ++ err,
    None => ()
  }

  result
}

/// Escape special characters in JSON strings
fn escape_json_string(s: String) -> String {
  let mut result = ""
  let mut idx = 0

  while idx < s.length() {
    let c = s[idx]
    match c {
      '"' => result = result ++ "\\\""
      '\\' => result = result ++ "\\\\"
      '\n' => result = result ++ "\\n"
      '\r' => result = result ++ "\\r"
      '\t' => result = result ++ "\\t"
      _ => result = result ++ String::from_byte(c)
    }
    idx = idx + 1
  }

  result
}

/// Output log (placeholder - would use stdout/stderr FFI in production)
fn output_log(message: String) -> Unit {
  // TODO: Implement FFI to write to stdout/stderr
  // For now, this is a no-op placeholder
  ()
}

/// Get current timestamp in milliseconds
/// TODO: Implement FFI to get current time
fn current_timestamp_ms() -> Int64 {
  0
}

/// Convert Int64 to string
fn int64_to_string(i: Int64) -> String {
  // TODO: Implement proper Int64 to string conversion
  "0"
}

// Module-level default logger
let default_logger = Logger::default()

/// Convenience functions using the default logger
pub fn log_debug(message: String) -> Unit {
  default_logger.debug(message)
}

pub fn log_info(message: String) -> Unit {
  default_logger.info(message)
}

pub fn log_warn(message: String) -> Unit {
  default_logger.warn(message)
}

pub fn log_error(message: String) -> Unit {
  default_logger.error(message)
}

pub fn log_warn_with_error(message: String, error: String) -> Unit {
  default_logger.warn_with_error(message, error)
}

pub fn log_error_with_error(message: String, error: String) -> Unit {
  default_logger.error_with_error(message, error)
}
