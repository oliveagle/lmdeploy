//! Unit tests for embeddings functionality
//!
//! These tests verify the embeddings output correctness.
//! For integration tests with actual models, see the
//! embeddings verification script in docs/.

#[cfg(test)]
mod embeddings_tests {
    use lmdeploy_server::model::cpp_engine::TurboMindCEngine;
    use std::time::Instant;

    /// Helper function to get model path from environment or default
    fn get_model_path() -> String {
        std::env::var("MODEL_PATH").unwrap_or_else(|_| {
            "/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B".to_string()
        })
    }

    /// Test that embeddings have correct dimensions
    #[tokio::test]
    #[ignore] // Requires GPU and model
    async fn test_embeddings_dimension() {
        let model_path = get_model_path();
        let engine = TurboMindCEngine::new(&model_path)
            .await
            .expect("Engine should load");

        let text = "Hello, world!";
        let embedding = engine.embed(text, None).await;

        assert!(!embedding.is_empty(), "Embedding should not be empty");
        assert!(embedding.len() >= 768, "Embedding should have at least 768 dimensions");
    }

    /// Test that dimension truncation works correctly
    #[tokio::test]
    #[ignore] // Requires GPU and model
    async fn test_embeddings_truncation() {
        let model_path = get_model_path();
        let engine = TurboMindCEngine::new(&model_path)
            .await
            .expect("Engine should load");

        let text = "Hello, world!";
        let full_embedding = engine.embed(text, None).await;
        let truncated_256 = engine.embed(text, Some(256)).await;
        let truncated_128 = engine.embed(text, Some(128)).await;

        assert_eq!(truncated_256.len(), 256, "Truncated to 256 dimensions");
        assert_eq!(truncated_128.len(), 128, "Truncated to 128 dimensions");

        // Verify truncated values match first N values of full embedding
        assert!(
            full_embedding[..256]
                .iter()
                .zip(truncated_256.iter())
                .all(|(a, b)| (a - b).abs() < f32::EPSILON),
            "Truncated values should match first N values"
        );
        assert!(
            full_embedding[..128]
                .iter()
                .zip(truncated_128.iter())
                .all(|(a, b)| (a - b).abs() < f32::EPSILON),
            "Truncated values should match first N values"
        );
    }

    /// Test that embeddings are deterministic
    #[tokio::test]
    #[ignore] // Requires GPU and model
    async fn test_embeddings_determinism() {
        let model_path = get_model_path();
        let engine = TurboMindCEngine::new(&model_path)
            .await
            .expect("Engine should load");

        let text = "The quick brown fox jumps over the lazy dog.";

        // Generate embeddings multiple times
        let emb1 = engine.embed(text, None).await;
        let emb2 = engine.embed(text, None).await;
        let emb3 = engine.embed(text, None).await;

        // All embeddings should be identical
        assert_eq!(emb1, emb2, "First and second embeddings should be identical");
        assert_eq!(emb2, emb3, "Second and third embeddings should be identical");
    }

    /// Test that different texts produce different embeddings
    #[tokio::test]
    #[ignore] // Requires GPU and model
    async fn test_embeddings_uniqueness() {
        let model_path = get_model_path();
        let engine = TurboMindCEngine::new(&model_path)
            .await
            .expect("Engine should load");

        let text1 = "The cat is sleeping.";
        let text2 = "The dog is running.";
        let text3 = "Machine learning is fascinating.";

        let emb1 = engine.embed(text1, None).await;
        let emb2 = engine.embed(text2, None).await;
        let emb3 = engine.embed(text3, None).await;

        // Calculate cosine similarities
        let cos_sim_12 = cosine_similarity(&emb1, &emb2);
        let cos_sim_13 = cosine_similarity(&emb1, &emb3);
        let cos_sim_23 = cosine_similarity(&emb2, &emb3);

        // Similar texts (cat/dog) should be somewhat similar but not identical
        assert!(cos_sim_12 > 0.5, "Related texts should have cosine similarity > 0.5");
        assert!(cos_sim_12 < 0.99, "Different texts should not be identical");

        // Unrelated texts should be less similar
        assert!(cos_sim_13 < 0.95, "Unrelated texts should be less similar");
        assert!(cos_sim_23 < 0.95, "Unrelated texts should be less similar");
    }

    /// Test empty input handling
    #[tokio::test]
    #[ignore] // Requires GPU and model
    async fn test_embeddings_empty_input() {
        let model_path = get_model_path();
        let engine = TurboMindCEngine::new(&model_path)
            .await
            .expect("Engine should load");

        let empty_embedding = engine.embed("", None).await;
        assert!(empty_embedding.is_empty(), "Empty input should produce empty embedding");

        let whitespace_embedding = engine.embed("   ", None).await;
        // After tokenization, whitespace might be empty or produce minimal tokens
        // This is expected behavior
    }

    /// Test embeddings performance
    #[tokio::test]
    #[ignore] // Requires GPU and model
    async fn test_embeddings_performance() {
        let model_path = get_model_path();
        let engine = TurboMindCEngine::new(&model_path)
            .await
            .expect("Engine should load");

        let text = "A quick brown fox jumps over the lazy dog.";
        let iterations = 10;

        let start = Instant::now();
        for _ in 0..iterations {
            let _embedding = engine.embed(text, None).await;
        }
        let elapsed = start.elapsed();

        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
        println!("Average embedding time: {:.2}ms", avg_ms);

        // Embeddings should be reasonably fast (adjust threshold as needed)
        assert!(
            avg_ms < 1000.0,
            "Embeddings should complete in under 1 second on average"
        );
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}
