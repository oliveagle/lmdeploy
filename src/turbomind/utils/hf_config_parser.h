// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <map>
#include <string>
#include <vector>
#include <any>
#include <sstream>
#include <cctype>
#include <memory>
#include <fstream>

namespace turbomind {

/// Simple JSON parser for HuggingFace config.json files
///
/// This is a minimal JSON parser that handles the subset of JSON
/// features used by HuggingFace config.json files:
/// - Objects (key-value pairs)
/// - Arrays
/// - Strings, numbers, booleans, null
/// - Nested objects (e.g., text_config, quantization_config)
///
/// No external dependencies - header-only implementation.
class HfConfigParser {
public:
    /// Supported JSON value types
    enum class Type {
        kNull,
        kBool,
        kInt,
        kFloat,
        kString,
        kObject,
        kArray
    };

    /// JSON value that can hold any type
    class Value {
    public:
        Value() : type_(Type::kNull) {}
        Value(bool v) : type_(Type::kBool), bool_val_(v) {}
        Value(int v) : type_(Type::kInt), int_val_(v) {}
        Value(int64_t v) : type_(Type::kInt), int_val_(v) {}
        Value(double v) : type_(Type::kFloat), float_val_(v) {}
        Value(const char* v) : type_(Type::kString), string_val_(v) {}
        Value(const std::string& v) : type_(Type::kString), string_val_(v) {}

        // Copy constructor
        Value(const Value& other)
            : type_(other.type_),
              bool_val_(other.bool_val_),
              int_val_(other.int_val_),
              float_val_(other.float_val_),
              string_val_(other.string_val_) {
            if (other.object_val_) {
                object_val_ = std::make_unique<std::map<std::string, Value>>(*other.object_val_);
            }
            if (other.array_val_) {
                array_val_ = std::make_unique<std::vector<Value>>(*other.array_val_);
            }
        }

        // Move constructor
        Value(Value&& other) noexcept
            : type_(other.type_),
              bool_val_(other.bool_val_),
              int_val_(other.int_val_),
              float_val_(other.float_val_),
              string_val_(std::move(other.string_val_)),
              object_val_(std::move(other.object_val_)),
              array_val_(std::move(other.array_val_)) {
            other.type_ = Type::kNull;
        }

        // Copy assignment
        Value& operator=(const Value& other) {
            if (this != &other) {
                type_ = other.type_;
                bool_val_ = other.bool_val_;
                int_val_ = other.int_val_;
                float_val_ = other.float_val_;
                string_val_ = other.string_val_;
                if (other.object_val_) {
                    object_val_ = std::make_unique<std::map<std::string, Value>>(*other.object_val_);
                } else {
                    object_val_.reset();
                }
                if (other.array_val_) {
                    array_val_ = std::make_unique<std::vector<Value>>(*other.array_val_);
                } else {
                    array_val_.reset();
                }
            }
            return *this;
        }

        // Move assignment
        Value& operator=(Value&& other) noexcept {
            if (this != &other) {
                type_ = other.type_;
                bool_val_ = other.bool_val_;
                int_val_ = other.int_val_;
                float_val_ = other.float_val_;
                string_val_ = std::move(other.string_val_);
                object_val_ = std::move(other.object_val_);
                array_val_ = std::move(other.array_val_);
                other.type_ = Type::kNull;
            }
            return *this;
        }

        Type type() const { return type_; }

        bool is_null() const { return type_ == Type::kNull; }
        bool is_bool() const { return type_ == Type::kBool; }
        bool is_int() const { return type_ == Type::kInt; }
        bool is_float() const { return type_ == Type::kFloat; }
        bool is_string() const { return type_ == Type::kString; }
        bool is_object() const { return type_ == Type::kObject; }
        bool is_array() const { return type_ == Type::kArray; }

        bool as_bool(bool default_val = false) const {
            return type_ == Type::kBool ? bool_val_ : default_val;
        }

        int64_t as_int(int64_t default_val = 0) const {
            if (type_ == Type::kInt) return int_val_;
            if (type_ == Type::kFloat) return static_cast<int64_t>(float_val_);
            return default_val;
        }

        double as_float(double default_val = 0.0) const {
            if (type_ == Type::kFloat) return float_val_;
            if (type_ == Type::kInt) return static_cast<double>(int_val_);
            return default_val;
        }

        const std::string& as_string(const std::string& default_val = "") const {
            return type_ == Type::kString ? string_val_ : default_val;
        }

        const std::map<std::string, Value>& as_object() const {
            static const std::map<std::string, Value> empty;
            return type_ == Type::kObject ? *object_val_ : empty;
        }

        const std::vector<Value>& as_array() const {
            static const std::vector<Value> empty;
            return type_ == Type::kArray ? *array_val_ : empty;
        }

        std::map<std::string, Value>& as_object() {
            if (type_ != Type::kObject) {
                type_ = Type::kObject;
                object_val_ = std::make_unique<std::map<std::string, Value>>();
            }
            return *object_val_;
        }

        std::vector<Value>& as_array() {
            if (type_ != Type::kArray) {
                type_ = Type::kArray;
                array_val_ = std::make_unique<std::vector<Value>>();
            }
            return *array_val_;
        }

        /// Get nested object by path (e.g., "text_config.hidden_size")
        const Value& get(const std::string& path) const {
            static const Value null_value;
            size_t dot_pos = path.find('.');
            if (dot_pos == std::string::npos) {
                if (!is_object()) return null_value;
                auto it = object_val_->find(path);
                return it != object_val_->end() ? it->second : null_value;
            } else {
                std::string key = path.substr(0, dot_pos);
                std::string rest = path.substr(dot_pos + 1);
                if (!is_object()) return null_value;
                auto it = object_val_->find(key);
                if (it == object_val_->end() || !it->second.is_object()) return null_value;
                return it->second.get(rest);
            }
        }

        /// Check if key exists in object
        bool has(const std::string& key) const {
            if (!is_object()) return false;
            return object_val_->find(key) != object_val_->end();
        }

        /// Array indexing
        const Value& operator[](size_t index) const {
            static const Value null_value;
            if (!is_array() || index >= array_val_->size()) return null_value;
            return (*array_val_)[index];
        }

        /// Object key access
        const Value& operator[](const std::string& key) const {
            return get(key);
        }

    private:
        Type type_;
        bool bool_val_ = false;
        int64_t int_val_ = 0;
        double float_val_ = 0.0;
        std::string string_val_;
        std::unique_ptr<std::map<std::string, Value>> object_val_;
        std::unique_ptr<std::vector<Value>> array_val_;
    };

    /// Parse JSON string
    /// Returns null value on error
    static Value Parse(const std::string& json) {
        Parser p(json);
        return p.ParseValue();
    }

    /// Parse config.json file
    /// Returns null value on error
    static Value ParseFile(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            return Value{};
        }
        std::string content((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
        return Parse(content);
    }

private:
    class Parser {
    public:
        explicit Parser(const std::string& json) : json_(json), pos_(0) {
            SkipWhitespace();
        }

        Value ParseValue() {
            SkipWhitespace();
            if (pos_ >= json_.size()) {
                return Value{};
            }

            char c = json_[pos_];
            if (c == '{') return ParseObject();
            if (c == '[') return ParseArray();
            if (c == '"') return ParseString();
            if (c == 't' || c == 'f') return ParseBool();
            if (c == 'n') return ParseNull();
            if (c == '-' || std::isdigit(c)) return ParseNumber();

            return Value{};
        }

    private:
        void SkipWhitespace() {
            while (pos_ < json_.size() &&
                   (json_[pos_] == ' ' || json_[pos_] == '\t' ||
                    json_[pos_] == '\n' || json_[pos_] == '\r')) {
                ++pos_;
            }
        }

        Value ParseObject() {
            Value result;
            result.as_object();  // Initialize as object

            ++pos_;  // Skip '{'
            SkipWhitespace();

            if (pos_ < json_.size() && json_[pos_] == '}') {
                ++pos_;  // Empty object
                return result;
            }

            while (pos_ < json_.size()) {
                SkipWhitespace();

                // Parse key (must be string)
                if (pos_ >= json_.size() || json_[pos_] != '"') {
                    return Value{};
                }
                std::string key = ParseString().as_string();

                SkipWhitespace();

                // Skip ':'
                if (pos_ >= json_.size() || json_[pos_] != ':') {
                    return Value{};
                }
                ++pos_;
                SkipWhitespace();

                // Parse value
                Value value = ParseValue();
                result.as_object()[key] = value;

                SkipWhitespace();

                // Check for ',' or '}'
                if (pos_ >= json_.size()) {
                    return Value{};
                }
                if (json_[pos_] == '}') {
                    ++pos_;
                    break;
                }
                if (json_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                return Value{};
            }

            return result;
        }

        Value ParseArray() {
            Value result;
            auto& array = result.as_array();  // Initialize as array

            ++pos_;  // Skip '['
            SkipWhitespace();

            if (pos_ < json_.size() && json_[pos_] == ']') {
                ++pos_;  // Empty array
                return result;
            }

            while (pos_ < json_.size()) {
                SkipWhitespace();

                Value value = ParseValue();
                array.push_back(value);

                SkipWhitespace();

                // Check for ',' or ']'
                if (pos_ >= json_.size()) {
                    return Value{};
                }
                if (json_[pos_] == ']') {
                    ++pos_;
                    break;
                }
                if (json_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                return Value{};
            }

            return result;
        }

        Value ParseString() {
            if (pos_ >= json_.size() || json_[pos_] != '"') {
                return Value{};
            }

            ++pos_;  // Skip opening '"'
            std::string result;
            bool escape = false;

            while (pos_ < json_.size()) {
                char c = json_[pos_];
                ++pos_;

                if (escape) {
                    switch (c) {
                        case '"':  result += '"'; break;
                        case '\\': result += '\\'; break;
                        case '/':  result += '/'; break;
                        case 'b':  result += '\b'; break;
                        case 'f':  result += '\f'; break;
                        case 'n':  result += '\n'; break;
                        case 'r':  result += '\r'; break;
                        case 't':  result += '\t'; break;
                        case 'u': {
                            // Unicode escape (4 hex digits)
                            if (pos_ + 4 > json_.size()) {
                                return Value{};
                            }
                            std::string hex = json_.substr(pos_, 4);
                            pos_ += 4;
                            int codepoint = std::stoi(hex, nullptr, 16);
                            // For simplicity, just decode ASCII range
                            if (codepoint < 128) {
                                result += static_cast<char>(codepoint);
                            }
                            break;
                        }
                        default:
                            result += c;
                            break;
                    }
                    escape = false;
                } else if (c == '\\') {
                    escape = true;
                } else if (c == '"') {
                    return Value(result);
                } else {
                    result += c;
                }
            }

            return Value{};  // Unterminated string
        }

        Value ParseNumber() {
            size_t start = pos_;

            if (pos_ < json_.size() && json_[pos_] == '-') {
                ++pos_;
            }

            while (pos_ < json_.size() && std::isdigit(json_[pos_])) {
                ++pos_;
            }

            bool is_float = false;
            if (pos_ < json_.size() && json_[pos_] == '.') {
                is_float = true;
                ++pos_;
                while (pos_ < json_.size() && std::isdigit(json_[pos_])) {
                    ++pos_;
                }
            }

            if (pos_ < json_.size() && (json_[pos_] == 'e' || json_[pos_] == 'E')) {
                is_float = true;
                ++pos_;
                if (pos_ < json_.size() && (json_[pos_] == '+' || json_[pos_] == '-')) {
                    ++pos_;
                }
                while (pos_ < json_.size() && std::isdigit(json_[pos_])) {
                    ++pos_;
                }
            }

            std::string num_str = json_.substr(start, pos_ - start);

            if (is_float) {
                return Value(std::stod(num_str));
            } else {
                return Value(static_cast<int64_t>(std::stoll(num_str)));
            }
        }

        Value ParseBool() {
            if (json_.substr(pos_, 4) == "true") {
                pos_ += 4;
                return Value(true);
            }
            if (json_.substr(pos_, 5) == "false") {
                pos_ += 5;
                return Value(false);
            }
            return Value{};
        }

        Value ParseNull() {
            if (json_.substr(pos_, 4) == "null") {
                pos_ += 4;
                return Value();
            }
            return Value{};
        }

        const std::string& json_;
        size_t pos_;
    };
};

}  // namespace turbomind
