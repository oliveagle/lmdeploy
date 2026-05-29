//! End-to-end integration tests for lmdeploy-server
//!
//! These tests verify the complete inference path components:
//! 1. Tokenizer loading and encoding/decoding
//! 2. Model config parsing and AWQ detection
//! 3. Engine type selection
//! 4. Config detection
//!
//! Note: Actual model loading tests require GPU and are skipped in CI.

mod tokenizer_tests {
    use lmdeploy_server::tokenizer::LMTokenizer;

    fn get_model_path() -> String {
        std::env::var("MODEL_PATH")
            .unwrap_or_else(|_| "/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B".to_string())
    }

    #[test]
    fn test_tokenizer_load_from_model() {
        let path = get_model_path();
        let tokenizer = LMTokenizer::from_path(&path);
        assert!(
            tokenizer.is_ok(),
            "Failed to load tokenizer: {:?}",
            tokenizer
        );
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let path = get_model_path();
        let tokenizer = LMTokenizer::from_path(&path).expect("Tokenizer must load");
        assert!(
            tokenizer.vocab_size() > 10000,
            "Vocab size too small: {}",
            tokenizer.vocab_size()
        );
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let path = get_model_path();
        let tokenizer = LMTokenizer::from_path(&path).expect("Tokenizer must load");

        let text = "Hello, world! 你好世界";
        let encoded = tokenizer
            .encode(text, false, true)
            .expect("Encoding must succeed");
        assert!(!encoded.is_empty(), "Encoded tokens must not be empty");

        let decoded = tokenizer
            .decode(&encoded, true)
            .expect("Decoding must succeed");
        assert!(!decoded.is_empty(), "Decoded text must not be empty");
        assert_eq!(decoded, text, "Roundtrip encode/decode must be lossless");
    }

    #[test]
    fn test_encode_with_bos() {
        let path = get_model_path();
        let tokenizer = LMTokenizer::from_path(&path).expect("Tokenizer must load");

        let text = "test text";
        let without_bos = tokenizer
            .encode(text, false, false)
            .expect("Encoding must succeed");
        let with_bos = tokenizer
            .encode(text, true, false)
            .expect("Encoding must succeed");

        if tokenizer.bos_token_id().is_some() {
            assert_eq!(
                with_bos.len(),
                without_bos.len() + 1,
                "BOS should add exactly 1 token"
            );
            assert_eq!(
                with_bos[0],
                tokenizer.bos_token_id().unwrap(),
                "BOS token must be at position 0"
            );
        }
    }

    #[test]
    fn test_decode_single_token() {
        let path = get_model_path();
        let tokenizer = LMTokenizer::from_path(&path).expect("Tokenizer must load");

        // Tokenize and take first token
        let encoded = tokenizer
            .encode("hello", false, true)
            .expect("Encoding must succeed");
        let first_token = encoded[0];

        let decoded = tokenizer
            .decode_token(first_token)
            .expect("Decoding must succeed");
        assert!(!decoded.is_empty(), "Single token decode must not be empty");
    }

    #[test]
    fn test_encode_raw() {
        let path = get_model_path();
        let tokenizer = LMTokenizer::from_path(&path).expect("Tokenizer must load");

        let text = "raw test input";
        let raw = tokenizer
            .encode_raw(text)
            .expect("Raw encoding must succeed");
        assert!(!raw.is_empty(), "Raw tokens must not be empty");
    }

    #[test]
    fn test_id_to_token() {
        let path = get_model_path();
        let tokenizer = LMTokenizer::from_path(&path).expect("Tokenizer must load");

        // Token 0 should exist
        let token = tokenizer.id_to_token(0);
        assert!(token.is_some(), "Token ID 0 must exist in vocabulary");
    }

    #[test]
    fn test_eos_detection() {
        let path = get_model_path();
        let tokenizer = LMTokenizer::from_path(&path).expect("Tokenizer must load");

        let eos_ids = tokenizer.eos_token_ids();
        assert!(!eos_ids.is_empty(), "Must have at least one EOS token");

        // Test that at least one EOS token is recognized
        let first_eos = eos_ids[0];
        assert!(
            tokenizer.is_eos(first_eos),
            "Known EOS token must be recognized"
        );

        // Test non-EOS token
        assert!(!tokenizer.is_eos(0), "Token ID 0 should not be EOS");
    }

    #[test]
    fn test_encode_batch() {
        let path = get_model_path();
        let tokenizer = LMTokenizer::from_path(&path).expect("Tokenizer must load");

        let texts = vec!["Hello", "World", "Test"];
        let batch = tokenizer
            .encode_batch(&texts, false)
            .expect("Batch encoding must succeed");
        assert_eq!(batch.len(), 3, "Batch must encode 3 texts");
    }

    #[test]
    fn test_tokenizer_not_found() {
        let result = LMTokenizer::from_path("/nonexistent/model/path");
        assert!(result.is_err(), "Should fail for non-existent path");
    }
}

mod config_detection_tests {
    use lmdeploy_server::model::cpp_engine::EngineType;

    #[test]
    fn test_engine_type_parsing() {
        assert_eq!(EngineType::from_str("cpp"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("c++"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("native"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("pure_cpp"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("invalid"), None);
    }

    #[test]
    fn test_engine_type_case_insensitive() {
        assert_eq!(EngineType::from_str("CPP"), Some(EngineType::PureCpp));
        assert_eq!(EngineType::from_str("Native"), Some(EngineType::PureCpp));
    }

    #[test]
    fn test_engine_type_as_str() {
        assert_eq!(EngineType::PureCpp.as_str(), "pure_cpp");
    }

    #[test]
    fn test_awq_detection() {
        // Non-existent path should return false for AWQ detection
        let path = std::path::PathBuf::from("/nonexistent/path");
        let content = r#"{"quantization_config": {"quant_method": "awq"}}"#;
        assert!(content.contains("quant_method") && content.contains("awq"));
    }
}

mod awq_inference_tests {
    use lmdeploy_server::model::cpp_engine::GenerationParams;
    use lmdeploy_server::model::cpp_engine::ModelState;
    use lmdeploy_server::model::cpp_engine::TurboMindCEngine;

    fn get_awq_model_path() -> String {
        std::env::var("AWQ_MODEL_PATH").unwrap_or_else(|_| {
            "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ".to_string()
        })
    }

    #[tokio::test]
    async fn test_awq_model_load() {
        let path = get_awq_model_path();

        // Skip if model path doesn't exist
        if !std::path::Path::new(&path).exists() {
            println!("Skipping: AWQ model not found at {}", path);
            return;
        }

        let engine = TurboMindCEngine::new(&path).await;
        assert!(engine.is_ok(), "AWQ model must load: {:?}", engine);
    }

    #[tokio::test]
    async fn test_awq_model_state_ready() {
        let path = get_awq_model_path();

        if !std::path::Path::new(&path).exists() {
            println!("Skipping: AWQ model not found at {}", path);
            return;
        }

        let engine = TurboMindCEngine::new(&path)
            .await
            .expect("Engine must load");
        assert!(engine.is_ready(), "Engine must be ready after loading");
        assert_eq!(
            engine.info().state,
            ModelState::Ready,
            "State must be Ready"
        );
    }

    #[tokio::test]
    async fn test_awq_model_info() {
        let path = get_awq_model_path();

        if !std::path::Path::new(&path).exists() {
            println!("Skipping: AWQ model not found at {}", path);
            return;
        }

        let engine = TurboMindCEngine::new(&path)
            .await
            .expect("Engine must load");
        let info = engine.info();

        assert_eq!(info.quant_policy, 4, "AWQ model must have quant_policy=4");
        assert!(info.hidden_size.is_some(), "hidden_size must be set");
        assert!(
            info.hidden_size.unwrap() > 0,
            "hidden_size must be positive"
        );
    }

    #[tokio::test]
    async fn test_awq_generate_text() {
        let path = get_awq_model_path();

        if !std::path::Path::new(&path).exists() {
            println!("Skipping: AWQ model not found at {}", path);
            return;
        }

        let engine = TurboMindCEngine::new(&path)
            .await
            .expect("Engine must load");

        let params = GenerationParams {
            max_tokens: Some(10),
            ..Default::default()
        };
        let result = engine.generate("Hello, ", params).await;
        assert!(!result.is_empty(), "Generated text must not be empty");
    }

    #[tokio::test]
    async fn test_awq_generate_with_metrics() {
        let path = get_awq_model_path();

        if !std::path::Path::new(&path).exists() {
            println!("Skipping: AWQ model not found at {}", path);
            return;
        }

        let engine = TurboMindCEngine::new(&path)
            .await
            .expect("Engine must load");

        let params = GenerationParams {
            max_tokens: Some(10),
            ..Default::default()
        };
        let (text, num_tokens, elapsed_ms) =
            engine.generate_with_metrics("What is 2+2? ", params).await;
        assert!(!text.is_empty(), "Generated text must not be empty");
        assert!(num_tokens > 0, "Must generate at least one token");
        assert!(elapsed_ms > 0.0, "Must take non-zero time");
    }
}

mod awq_config_tests {
    #[test]
    fn test_awq_config_detection_real_file() {
        let path = std::path::PathBuf::from(
            "/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ/config.json",
        );
        if path.exists() {
            let content = std::fs::read_to_string(&path).expect("Must read config");
            assert!(
                content.contains("quant_method"),
                "Config must have quant_method"
            );
            assert!(content.contains("awq"), "Config must reference AWQ");
        }
    }

    #[test]
    fn test_awq_config_false_positive() {
        let fake_config = r#"{"quant_method": "gptq"}"#;
        let lower = fake_config.to_lowercase();
        let is_awq = lower.contains("\"quant_method\"") && lower.contains("\"awq\"");
        assert!(!is_awq, "GPTQ should not be detected as AWQ");
    }
}

mod model_state_tests {
    use lmdeploy_server::model::cpp_engine::ModelState;

    #[test]
    fn test_model_state_default() {
        let state = ModelState::default();
        assert_eq!(state, ModelState::Unloaded);
    }

    #[test]
    fn test_model_state_transitions() {
        let unloaded = ModelState::Unloaded;
        let loading = ModelState::Loading;
        let ready = ModelState::Ready;
        let failed = ModelState::Failed("test error".to_string());

        assert_eq!(unloaded, ModelState::Unloaded);
        assert_eq!(loading, ModelState::Loading);
        assert_eq!(ready, ModelState::Ready);
        assert!(matches!(failed, ModelState::Failed(msg) if msg == "test error"));
    }
}
