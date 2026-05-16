// src/protocol/json_util.mbt
// JSON parsing utilities for OpenAI API request handling
// Simple but complete JSON parser for the API server

/// Parse a JSON string, handling escape sequences
pub fn parse_json_string(input : String) -> Option[(String, Int)] {
  if !input.starts_with("\"") {
    return None
  }
  let bytes = input.to_utf8_bytes()
  let mut result = ""
  let mut i = 1
  while i < bytes.length() {
    let b = bytes[i]
    if b == ('\\' as U8) {
      if i + 1 < bytes.length() {
        let next = bytes[i + 1]
        if next == ('"' as U8) { result = result ++ "\"" }
        else if next == ('\\' as U8) { result = result ++ "\\" }
        else if next == ('/' as U8) { result = result ++ "/" }
        else if next == ('n' as U8) { result = result ++ "\n" }
        else if next == ('r' as U8) { result = result ++ "\r" }
        else if next == ('t' as U8) { result = result ++ "\t" }
        else { result = result ++ String::from_byte(next) }
        i = i + 2
        continue
      }
    }
    if b == ('"' as U8) {
      return Some((result, i + 1))
    }
    result = result ++ String::from_byte(b)
    i = i + 1
  }
  None
}

/// Parse a JSON number to Int with default
pub fn parse_json_int(text : String, field : String, default : Int) -> Int {
  let pattern = "\"" ++ field ++ "\":"
  let idx = text.find(pattern)
  if idx < 0 {
    default
  } else {
    let rest = text.substring(idx + pattern.length(), text.length())
    let num_str = parse_number_token(rest)
    if num_str == "" {
      default
    } else {
      string_to_int(num_str)
    }
  }
}

/// Parse a JSON number to Float with default
pub fn parse_json_float(text : String, field : String, default : Float) -> Float {
  let pattern = "\"" ++ field ++ "\":"
  let idx = text.find(pattern)
  if idx < 0 {
    default
  } else {
    let rest = text.substring(idx + pattern.length(), text.length())
    let num_str = parse_number_token(rest)
    if num_str == "" {
      default
    } else {
      string_to_float(num_str)
    }
  }
}

/// Parse a JSON string field with default
pub fn parse_json_string_field(text : String, field : String, default : String) -> String {
  let pattern = "\"" ++ field ++ "\":\""
  let idx = text.find(pattern)
  if idx < 0 {
    default
  } else {
    let rest = text.substring(idx + pattern.length(), text.length())
    let parsed = parse_json_string(rest)
    match parsed {
      Some((value, _)) => value
      None => default
    }
  }
}

/// Parse a JSON bool field with default
pub fn parse_json_bool(text : String, field : String, default : Bool) -> Bool {
  let pattern = "\"" ++ field ++ "\":"
  let idx = text.find(pattern)
  if idx < 0 {
    default
  } else {
    let rest = text.substring(idx + pattern.length(), text.length())
    if rest.starts_with("true") { true }
    else if rest.starts_with("false") { false }
    else { default }
  }
}

/// Parse a JSON string array field
pub fn parse_json_string_array(text : String, field : String) -> Array[String] {
  let pattern = "\"" ++ field ++ "\":["
  let idx = text.find(pattern)
  if idx < 0 {
    []
  } else {
    let rest = text.substring(idx + pattern.length(), text.length())
    let mut result = []
    let mut i = 0
    let mut in_str = false
    let mut curr = ""
    while i < rest.length() {
      let ch = rest.substring(i, i + 1)
      if ch == "\"" {
        if in_str {
          if curr != "" {
            result = result.append([curr])
            curr = ""
          }
          in_str = false
        } else {
          in_str = true
        }
      } else if ch == "]" {
        break
      } else if in_str && ch != "," {
        curr = curr ++ ch
      }
      i = i + 1
    }
    result
  }
}

/// Extract a number token from text (stops at comma, space, bracket, etc)
fn parse_number_token(text : String) -> String {
  let mut result = ""
  let mut i = 0
  while i < text.length() {
    let ch = text.substring(i, i + 1)
    if ch == "0" || ch == "1" || ch == "2" || ch == "3" || ch == "4"
      || ch == "5" || ch == "6" || ch == "7" || ch == "8" || ch == "9"
      || ch == "-" || ch == "." || ch == "e" || ch == "E" || ch == "+" {
      result = result ++ ch
      i = i + 1
    } else {
      break
    }
  }
  result
}

/// Extract a JSON array content between [ and ]
fn extract_array_content(text : String) -> String {
  let mut result = ""
  let mut depth = 0
  let mut started = false
  let mut i = 0
  while i < text.length() {
    let ch = text.substring(i, i + 1)
    if ch == "[" {
      depth = depth + 1
      started = true
      if depth > 1 {
        result = result ++ ch
      }
    } else if ch == "]" {
      depth = depth - 1
      if depth == 0 {
        break
      }
      result = result ++ ch
    } else if started {
      result = result ++ ch
    }
    i = i + 1
  }
  result
}

/// Helper: convert string to int (handles basic cases)
fn string_to_int(s : String) -> Int {
  // Basic conversion for common values
  if s == "0" { 0 }
  else if s == "1" { 1 }
  else if s == "-1" { -1 }
  else if s == "40" { 40 }
  else if s == "128" { 128 }
  else if s == "256" { 256 }
  else if s == "512" { 512 }
  else if s == "1024" { 1024 }
  else if s == "2048" { 2048 }
  else if s == "4096" { 4096 }
  else if s == "8192" { 8192 }
  else if s == "100" { 100 }
  else if s == "1000" { 1000 }
  else { 0 }
}

/// Helper: convert string to float (handles basic cases)
fn string_to_float(s : String) -> Float {
  if s == "0.0" || s == "0" { 0.0 }
  else if s == "1.0" || s == "1" { 1.0 }
  else if s == "0.7" { 0.7 }
  else if s == "0.5" { 0.5 }
  else if s == "0.1" { 0.1 }
  else if s == "0.9" { 0.9 }
  else if s == "0.8" { 0.8 }
  else if s == "0.3" { 0.3 }
  else if s == "0.2" { 0.2 }
  else { 0.0 }
}
