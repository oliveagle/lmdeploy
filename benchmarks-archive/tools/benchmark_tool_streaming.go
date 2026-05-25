package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// Config 配置结构
type Config struct {
	BaseURL      string   `json:"base_url"`
	Model        string   `json:"model"`
	ContextSizes []int    `json:"context_sizes"`
	OutputLens   []int    `json:"output_lens"`
	Concurrency  []int    `json:"concurrency"`
	NumRequests  int      `json:"num_requests"`
	APIKey       string   `json:"api_key,omitempty"`
}

// BenchmarkResult 基准测试结果
type BenchmarkResult struct {
	Scenario               string
	ContextSize            int
	OutputLen              int
	Concurrency            int
	NumRequests            int
	TotalDuration          time.Duration
	PrefillDuration        time.Duration
	DecodeDuration         time.Duration
	TokensPerSecond        float64
	PrefillTokensPerSec    float64
	DecodeTokensPerSec  float64
	SuccessCount           int
	ErrorCount             int
	Error                  string
	FirstTokenLatencyMs    float64
	AvgTokenLatencyMs      float64
	P99TokenLatencyMs      float64
}

// StreamingRequest 流式请求
type StreamingRequest struct {
	Model    string          `json:"model"`
	Messages []OpenAIMessage `json:"messages"`
	MaxTokens int            `json:"max_tokens"`
	Stream    bool            `json:"stream"`
	Temperature float64       `json:"temperature"`
}

// OpenAIMessage 消息格式
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// generatePrompt 生成指定 token 数量的 prompt
func generatePrompt(targetTokens int) string {
	baseText := "This is a benchmark test for measuring LLM inference performance. " +
			"The quick brown fox jumps over the lazy dog. " +
			"Lorem ipsum dolor sit amet, consectetur adipiscing elit. " +
			"Please write a detailed explanation about computer science and artificial intelligence. "

	estimatedChars := targetTokens * 4
	repeat := estimatedChars / len(baseText) + 1

	prompt := strings.Repeat(baseText, repeat)

	if len(prompt) > estimatedChars {
		prompt = prompt[:estimatedChars]
	}

	return prompt
}

// sendStreamingRequest 发送流式请求并测量时间
func sendStreamingRequest(client *http.Client, config Config, prompt string, maxTokens int, contextSize int) (*BenchmarkResult, error) {
	reqBody := StreamingRequest{
		Model:       config.Model,
		Messages:    []OpenAIMessage{{Role: "user", Content: prompt}},
		MaxTokens:   maxTokens,
		Stream:      true,
		Temperature: 0.7,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/v1/chat/completions", config.BaseURL)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+config.APIKey)
	}

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	// Process streaming response
	br := bufio.NewReader(resp.Body)
	firstTokenTime := time.Time{}
	var completionTokens int
	var tokenCount int // Track actual tokens received

	for {
		line, err := br.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if strings.TrimSpace(data) == "[DONE]" {
			break
		}

		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		// Extract usage.completion_tokens if present (final chunk)
		if usage, ok := chunk["usage"].(map[string]interface{}); ok {
			if ct, ok := usage["completion_tokens"].(float64); ok {
				completionTokens = int(ct)
			}
		}

		choices, ok := chunk["choices"].([]interface{})
		if !ok || len(choices) == 0 {
			continue
		}

		choice := choices[0].(map[string]interface{})
		delta, hasDelta := choice["delta"].(map[string]interface{})
		if !hasDelta {
			continue
		}

		// Check if this chunk has content (not just finish_reason)
		if content, ok := delta["content"].(string); ok && content != "" {
			tokenCount++
		}

		// Track first token time
		if firstTokenTime.IsZero() && tokenCount > 0 {
			firstTokenTime = time.Now()
		}
	}
	resp.Body.Close()
	totalDuration := time.Since(start)

	// Use actual token count if completion_tokens is 0
	if completionTokens == 0 && tokenCount > 0 {
		completionTokens = tokenCount
	}

	// Calculate metrics
	result := &BenchmarkResult{
		TotalDuration:     totalDuration,
		PrefillDuration:   time.Duration(0),
		DecodeDuration:    time.Duration(0),
		TokensPerSecond:   float64(completionTokens) / totalDuration.Seconds(),
		SuccessCount:      1,
	}

	if !firstTokenTime.IsZero() {
		result.PrefillDuration = firstTokenTime.Sub(start)
		result.FirstTokenLatencyMs = float64(result.PrefillDuration.Milliseconds())
		result.DecodeDuration = totalDuration - result.PrefillDuration

		// Use actual completion tokens from usage
		if result.PrefillDuration > 0 {
			result.PrefillTokensPerSec = float64(contextSize) / result.PrefillDuration.Seconds()
		}
		if result.DecodeDuration > 0 && completionTokens > 0 {
			result.DecodeTokensPerSec = float64(completionTokens) / result.DecodeDuration.Seconds()
		}
	}

	return result, nil
}

// runBenchmark 运行单个基准测试场景
func runBenchmark(config Config, contextSize, outputLen, concurrency int) BenchmarkResult {
	result := BenchmarkResult{
		Scenario:    fmt.Sprintf("C%d_O%d", contextSize, outputLen),
		ContextSize: contextSize,
		OutputLen:   outputLen,
		Concurrency: concurrency,
		NumRequests: config.NumRequests,
	}

	prompt := generatePrompt(contextSize)
	client := &http.Client{Timeout: 5 * time.Minute}

	var wg sync.WaitGroup
	results := make(chan *BenchmarkResult, config.NumRequests)
	errors := make(chan error, config.NumRequests)

	startTime := time.Now()

	// Worker pool
	requestsPerWorker := config.NumRequests / concurrency
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < requestsPerWorker; j++ {
				resp, err := sendStreamingRequest(client, config, prompt, outputLen, contextSize)
				if err != nil {
					errors <- err
					results <- nil
					continue
				}
				results <- resp
			}
		}()
	}

	// Wait for completion
	wg.Wait()
	result.TotalDuration = time.Since(startTime)

	// Collect results
	var validResults []*BenchmarkResult
	for i := 0; i < config.NumRequests; i++ {
		select {
		case r := <-results:
			if r != nil {
				validResults = append(validResults, r)
			}
		case <-errors:
			result.ErrorCount++
		}
	}

	result.SuccessCount = len(validResults)

	if len(validResults) > 0 {
		// Average the metrics
		var sumPrefill, sumDecode time.Duration
		var sumTokensPerSec, sumPrefillTPS, sumDecodeTPS float64
		var sumFirstToken, sumAvgToken, sumP99Token float64

		for _, r := range validResults {
			sumPrefill += r.PrefillDuration
			sumDecode += r.DecodeDuration
			sumTokensPerSec += r.TokensPerSecond
			sumPrefillTPS += r.PrefillTokensPerSec
			sumDecodeTPS += r.DecodeTokensPerSec
			sumFirstToken += r.FirstTokenLatencyMs
			sumAvgToken += r.AvgTokenLatencyMs
			sumP99Token += r.P99TokenLatencyMs
		}

		n := float64(len(validResults))
		result.PrefillDuration = time.Duration(int64(float64(sumPrefill.Milliseconds()) / n) * 1e6)
		result.DecodeDuration = time.Duration(int64(float64(sumDecode.Milliseconds()) / n) * 1e6)
		result.TokensPerSecond = sumTokensPerSec / n
		result.PrefillTokensPerSec = sumPrefillTPS / n
		result.DecodeTokensPerSec = sumDecodeTPS / n
		result.FirstTokenLatencyMs = sumFirstToken / n
		result.AvgTokenLatencyMs = sumAvgToken / n
		result.P99TokenLatencyMs = sumP99Token / n
	}

	return result
}

// printStats 打印统计表格
func printStats(results []BenchmarkResult) {
	scenarios := make(map[string][]BenchmarkResult)
	for _, r := range results {
		scenarios[r.Scenario] = append(scenarios[r.Scenario], r)
	}

	fmt.Println("\n" + strings.Repeat("=", 140))
	fmt.Println("LMDeploy Benchmark Results - Streaming Mode")
	fmt.Println(strings.Repeat("=", 140))

	// Detailed results table
	fmt.Println("\n=== DETAILED RESULTS ===")
	fmt.Printf("%-15s | %-10s | %-8s | %-10s | %-15s | %-15s | %-15s | %-12s | %-12s\n",
		"Scenario", "Context", "Output", "Concurrency", "FirstToken(ms)", "AvgToken(ms)", "P99Token(ms)", "Prefillt/s", "Decodet/s")
	fmt.Println(strings.Repeat("-", 140))

	for _, r := range results {
		if r.SuccessCount > 0 {
			fmt.Printf("%-15s | %-10d | %-8d | %-10d | %-15.2f | %-15.2f | %-15.2f | %-12.0f | %-12.2f\n",
				r.Scenario, r.ContextSize, r.OutputLen, r.Concurrency,
				r.FirstTokenLatencyMs, r.AvgTokenLatencyMs, r.P99TokenLatencyMs,
				r.PrefillTokensPerSec, r.DecodeTokensPerSec)
		}
	}

	// Summary table by concurrency
	fmt.Println("\n=== THROUGHPUT SUMMARY (tokens/second) ===")
	fmt.Printf("%-15s", "Scenario")
	for _, c := range []int{1, 2, 4} {
		fmt.Printf(" | Concurrency=%-8d", c)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 100))

	for _, scenario := range sortedKeys(scenarios) {
		fmt.Printf("%-15s", scenario)
		for _, c := range []int{1, 2, 4} {
			found := false
			for _, r := range scenarios[scenario] {
				if r.Concurrency == c && r.DecodeTokensPerSec > 0 {
					fmt.Printf(" | %-18.2f", r.DecodeTokensPerSec)
					found = true
					break
				}
			}
			if !found {
				fmt.Printf(" | %-18s", "N/A")
			}
		}
		fmt.Println()
	}

	// First token latency summary
	fmt.Println("\n=== FIRST TOKEN LATENCY (ms) ===")
	fmt.Printf("%-15s", "Scenario")
	for _, c := range []int{1, 2, 4} {
		fmt.Printf(" | Concurrency=%-8d", c)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 100))

	for _, scenario := range sortedKeys(scenarios) {
		fmt.Printf("%-15s", scenario)
		for _, c := range []int{1, 2, 4} {
			found := false
			for _, r := range scenarios[scenario] {
				if r.Concurrency == c && r.FirstTokenLatencyMs > 0 {
					fmt.Printf(" | %-18.2f", r.FirstTokenLatencyMs)
					found = true
					break
				}
			}
			if !found {
				fmt.Printf(" | %-18s", "N/A")
			}
		}
		fmt.Println()
	}
}

func sortedKeys(m map[string][]BenchmarkResult) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func main() {
	configFile := flag.String("config", "", "Configuration file (JSON)")
	baseURL := flag.String("url", "http://localhost:8000", "LMDeploy server base URL")
	model := flag.String("model", "", "Model name (default: use path)")
	contextSizes := flag.String("contexts", "512,1024,2048,4096", "Comma-separated context sizes to test")
	outputLens := flag.String("outputs", "128,256,512", "Comma-separated output lengths to test")
	concurrency := flag.String("concurrency", "1,2,4", "Comma-separated concurrency levels to test")
	numRequests := flag.Int("requests", 10, "Number of requests per scenario")
	apiKey := flag.String("api-key", "", "API key for authentication")
	outputFile := flag.String("output", "", "Output file for JSON results")
	flag.Parse()

	var config Config

	if *configFile != "" {
		data, err := os.ReadFile(*configFile)
		if err != nil {
			log.Fatalf("Failed to read config file: %v", err)
		}
		if err := json.Unmarshal(data, &config); err != nil {
			log.Fatalf("Failed to parse config file: %v", err)
		}
	} else {
		config.BaseURL = *baseURL
		config.Model = *model
		config.APIKey = *apiKey
		config.NumRequests = *numRequests

		for _, s := range strings.Split(*contextSizes, ",") {
			var size int
			fmt.Sscanf(strings.TrimSpace(s), "%d", &size)
			config.ContextSizes = append(config.ContextSizes, size)
		}

		for _, s := range strings.Split(*outputLens, ",") {
			var len int
			fmt.Sscanf(strings.TrimSpace(s), "%d", &len)
			config.OutputLens = append(config.OutputLens, len)
		}

		for _, s := range strings.Split(*concurrency, ",") {
			var c int
			fmt.Sscanf(strings.TrimSpace(s), "%d", &c)
			config.Concurrency = append(config.Concurrency, c)
		}
	}

	fmt.Printf("LMDeploy Benchmark Tool (Streaming Mode)\n")
	fmt.Printf("======================================\n")
	fmt.Printf("Server URL: %s\n", config.BaseURL)
	fmt.Printf("Model: %s\n", config.Model)
	fmt.Printf("Context sizes: %v\n", config.ContextSizes)
	fmt.Printf("Output lengths: %v\n", config.OutputLens)
	fmt.Printf("Concurrency levels: %v\n", config.Concurrency)
	fmt.Printf("Requests per scenario: %d\n", config.NumRequests)
	fmt.Printf("\n")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(config.BaseURL + "/v1/models")
	if err != nil {
		log.Fatalf("Failed to connect to server: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Fatalf("Server returned status: %d", resp.StatusCode)
	}
	fmt.Println("Server connection OK")

	var allResults []BenchmarkResult
	totalScenarios := len(config.ContextSizes) * len(config.OutputLens) * len(config.Concurrency)
	scenarioNum := 0

	startTime := time.Now()

	for _, contextSize := range config.ContextSizes {
		for _, outputLen := range config.OutputLens {
			for _, c := range config.Concurrency {
				scenarioNum++
				fmt.Printf("\n[%d/%d] Testing: context=%d, output=%d, concurrency=%d\n",
					scenarioNum, totalScenarios, contextSize, outputLen, c)

				result := runBenchmark(config, contextSize, outputLen, c)
				allResults = append(allResults, result)

				if result.SuccessCount > 0 {
					fmt.Printf("  Success: %d/%d | FirstToken: %.2fms | Decode: %.2f t/s\n",
						result.SuccessCount, result.NumRequests,
						result.FirstTokenLatencyMs, result.DecodeTokensPerSec)
				} else {
					fmt.Printf("  FAILED: %s\n", result.Error)
				}
			}
		}
	}

	totalDuration := time.Since(startTime)

	printStats(allResults)

	fmt.Printf("\nTotal benchmark time: %v\n", totalDuration)

	if *outputFile != "" {
		data, err := json.MarshalIndent(allResults, "", "  ")
		if err != nil {
			log.Printf("Failed to marshal results: %v", err)
		} else {
			if err := os.WriteFile(*outputFile, data, 0644); err != nil {
				log.Printf("Failed to write output file: %v", err)
			} else {
				fmt.Printf("Results saved to: %s\n", *outputFile)
			}
		}
	}
}
