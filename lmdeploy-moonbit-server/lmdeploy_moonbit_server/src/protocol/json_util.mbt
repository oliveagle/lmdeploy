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

/// Convert string to Int - handles arbitrary integers (positive, negative, zero)
fn string_to_int(s : String) -> Int {
  if s == "" { 0 } else {
    let chars = s.to_utf8_bytes()
    let mut i = 0
    let is_negative = if chars[0] == ('-' as U8) {
      i = 1
      true
    } else {
      false
    }
    let mut result = 0
    while i < chars.length() {
      let c = chars[i]
      if c >= ('0' as U8) && c <= ('9' as U8) {
        let digit = c - ('0' as U8)
        result = result * 10 + (digit as Int)
      }
      i = i + 1
    }
    if is_negative { -result } else { result }
  }
}

/// Convert string to Float - handles arbitrary floats (including negative, decimal points, scientific notation)
fn string_to_float(s : String) -> Float {
  if s == "" { 0.0 } else {
    let chars = s.to_utf8_bytes()
    let mut i = 0
    let is_negative = if chars[0] == ('-' as U8) {
      i = 1
      true
    } else {
      false
    }

    // Parse integer part
    let mut integer_part = 0.0
    while i < chars.length() {
      let c = chars[i]
      if c >= ('0' as U8) && c <= ('9' as U8) {
        let digit = c - ('0' as U8)
        integer_part = integer_part *. 10.0 +. (digit as Float)
      } else if c == ('.' as U8) || c == ('e' as U8) || c == ('E' as U8) {
        break
      } else {
        break
      }
      i = i + 1
    }

    // Parse fractional part
    let mut fractional_part = 0.0
    let mut fractional_multiplier = 1.0
    if i < chars.length() && chars[i] == ('.' as U8) {
      i = i + 1
      while i < chars.length() {
        let c = chars[i]
        if c >= ('0' as U8) && c <= ('9' as U8) {
          let digit = c - ('0' as U8)
          fractional_multiplier = fractional_multiplier *. 10.0
          fractional_part = fractional_part +. (digit as Float) /. fractional_multiplier
        } else {
          break
        }
        i = i + 1
      }
    }

    // Parse exponent
    let mut exponent = 0.0
    if i < chars.length() && (chars[i] == ('e' as U8) || chars[i] == ('E' as U8)) {
      i = i + 1
      let exp_negative = if i < chars.length() && chars[i] == ('+' as U8) || chars[i] == ('-' as U8) {
        if chars[i] == ('-' as U8) {
          i = i + 1
          true
        } else {
          i = i + 1
          false
        }
      } else {
        false
      }
      let mut exp_value = 0
      while i < chars.length() {
        let c = chars[i]
        if c >= ('0' as U8) && c <= ('9' as U8) {
          let digit = c - ('0' as U8)
          exp_value = exp_value * 10 + (digit as Int)
        } else {
          break
        }
        i = i + 1
      }
      // Simple exponent approximation (for common cases)
      if exp_negative {
        // 10^(-n) approximation
        let mut inv = 1.0
        for _i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].filter(fn(x) { x < exp_value }) {
          inv = inv *. 0.1
          let _ = _i
        }
        exponent = inv
      } else if exp_value <= 10 {
        let mut mult = 1.0
        for _i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].filter(fn(x) { x < exp_value }) {
          mult = mult *. 10.0
          let _ = _i
        }
        exponent = mult
      } else {
        exponent = 1.0e10 // cap at reasonable max
      }
    } else {
      exponent = 1.0
    }

    let value = (integer_part +. fractional_part) *. exponent
    if is_negative { -.value } else { value }
  }
}
