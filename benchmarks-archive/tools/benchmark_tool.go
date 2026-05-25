package main

import (
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
	BaseURL     string   `json:"base_url"`
	Model       string   `json:"model"`
	ContextSizes []int   `json:"context_sizes"`     // 测试的上下文大小
	OutputLens  []int    `json:"output_lens"`       // 测试的输出长度
	Concurrency []int    `json:"concurrency"`       // 测试的并发数
	NumRequests int      `json:"num_requests"`      // 每个场景的请求数
	APIKey      string   `json:"api_key,omitempty"`
}

// BenchmarkResult 基准测试结果
type BenchmarkResult struct {
	Scenario          string
	ContextSize       int
	OutputLen         int
	Concurrency       int
	NumRequests       int
	TotalDuration     time.Duration
	PrefillDuration   time.Duration
	DecodeDuration    time.Duration
	TokensPerSecond   float64
	PrefillTokensPerSec float64
	DecodeTokensPerSec  float64
	SuccessCount      int
	ErrorCount        int
	Error             string
}

// Stats 统计信息
type Stats struct {
	Min   time.Duration
	Max   time.Duration
	Mean  time.Duration
	P50   time.Duration
	P95   time.Duration
	P99   time.Duration
}

// OpenAIRequest OpenAI API 请求格式
type OpenAIRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
}

// OpenAIMessage 消息格式
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OpenAIResponse OpenAI API 响应格式
type OpenAIResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// generatePrompt 生成指定 token 数量的 prompt
func generatePrompt(targetTokens int) string {
	baseText := "This is a benchmark test for measuring LLM inference performance. " +
		"The quick brown fox jumps over the lazy dog. " +
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit. " +
		"Please write a detailed explanation about computer science and artificial intelligence. "

	// 估算：每个 token 约 4 个字符
	estimatedChars := targetTokens * 4
	repeat := estimatedChars / len(baseText) + 1

	prompt := strings.Repeat(baseText, repeat)

	// 截断到大约目标长度
	if len(prompt) > estimatedChars {
		prompt = prompt[:estimatedChars]
	}

	return prompt
}

// sendRequest 发送单个请求并测量时间
func sendRequest(client *http.Client, config Config, prompt string, maxTokens int) (*OpenAIResponse, time.Duration, error) {
	reqBody := OpenAIRequest{
		Model: config.Model,
		Messages: []OpenAIMessage{
			{Role: "user", Content: prompt},
		},
		MaxTokens:   maxTokens,
		Stream:      false,
		Temperature: 0.7,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, 0, err
	}

	url := fmt.Sprintf("%s/v1/chat/completions", config.BaseURL)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, 0, err
	}

	req.Header.Set("Content-Type", "application/json")
	if config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+config.APIKey)
	}

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	duration := time.Since(start)

	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, duration, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, duration, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var openAIResp OpenAIResponse
	if err := json.Unmarshal(body, &openAIResp); err != nil {
		return nil, duration, err
	}

	return &openAIResp, duration, nil
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
	requests := make(chan int, config.NumRequests)
	results := make(chan time.Duration, config.NumRequests)
	errors := make(chan error, config.NumRequests)

	// Worker pool
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range requests {
				resp, duration, err := sendRequest(client, config, prompt, outputLen)
				if err != nil {
					errors <- err
					results <- 0
					continue
				}
				results <- duration
				_ = resp // Avoid unused variable warning
			}
		}()
	}

	startTime := time.Now()

	// 发送请求
	for i := 0; i < config.NumRequests; i++ {
		requests <- i
	}
	close(requests)

	// 收集结果
	var durations []time.Duration
	for i := 0; i < config.NumRequests; i++ {
		select {
		case d := <-results:
			if d > 0 {
				durations = append(durations, d)
			}
		case <-errors:
			result.ErrorCount++
		}
	}

	wg.Wait()
	result.TotalDuration = time.Since(startTime)
	result.SuccessCount = len(durations)

	if len(durations) > 0 {
		sort.Slice(durations, func(i, j int) bool {
			return durations[i] < durations[j]
		})

		// 计算统计信息
		sum := time.Duration(0)
		for _, d := range durations {
			sum += d
		}
		mean := sum / time.Duration(len(durations))

		// 估算性能指标
		// 假设：prefill 时间与 contextSize 成正比
		// decode 时间与 outputLen 成正比
		// 使用平均延迟来估算
		result.PrefillDuration = time.Duration(float64(mean) * float64(contextSize) / float64(contextSize+outputLen))
		result.DecodeDuration = mean - result.PrefillDuration

		// 计算 tokens/s
		totalTokens := contextSize + outputLen
		result.TokensPerSecond = float64(totalTokens) / mean.Seconds()
		result.PrefillTokensPerSec = float64(contextSize) / result.PrefillDuration.Seconds()
		result.DecodeTokensPerSec = float64(outputLen) / result.DecodeDuration.Seconds()
	}

	return result
}

// printStats 打印统计表格
func printStats(results []BenchmarkResult) {
	// 按 scenario 分组
	scenarios := make(map[string][]BenchmarkResult)
	for _, r := range results {
		scenarios[r.Scenario] = append(scenarios[r.Scenario], r)
	}

	fmt.Println("\n" + strings.Repeat("=", 120))
	fmt.Println("BENCHMARK RESULTS SUMMARY")
	fmt.Println(strings.Repeat("=", 120))

	// Prefill 性能表格
	fmt.Println("\n=== PREFILL PERFORMANCE (tokens/second) ===")
	fmt.Printf("%-20s", "Scenario")
	for _, c := range []int{1, 2, 4, 8} {
		fmt.Printf(" | Concurrency=%-8d", c)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 120))

	for _, scenario := range sortedKeys(scenarios) {
		fmt.Printf("%-20s", scenario)
		for _, c := range []int{1, 2, 4, 8} {
			found := false
			for _, r := range scenarios[scenario] {
				if r.Concurrency == c && r.PrefillTokensPerSec > 0 {
					fmt.Printf(" | %-18.0f", r.PrefillTokensPerSec)
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

	// Decode 性能表格
	fmt.Println("\n=== DECODE PERFORMANCE (tokens/second) ===")
	fmt.Printf("%-20s", "Scenario")
	for _, c := range []int{1, 2, 4, 8} {
		fmt.Printf(" | Concurrency=%-8d", c)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 120))

	for _, scenario := range sortedKeys(scenarios) {
		fmt.Printf("%-20s", scenario)
		for _, c := range []int{1, 2, 4, 8} {
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

	// 详细结果
	fmt.Println("\n=== DETAILED RESULTS ===")
	fmt.Printf("%-20s | %-12s | %-10s | %-12s | %-12s | %-12s | %-12s\n",
		"Scenario", "Context", "Output", "Concurrency", "Prefill t/s", "Decode t/s", "Overall t/s")
	fmt.Println(strings.Repeat("-", 120))

	for _, r := range results {
		if r.SuccessCount > 0 {
			fmt.Printf("%-20s | %-12d | %-10d | %-12d | %-12.0f | %-12.2f | %-12.2f\n",
				r.Scenario, r.ContextSize, r.OutputLen, r.Concurrency,
				r.PrefillTokensPerSec, r.DecodeTokensPerSec, r.TokensPerSecond)
		}
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
	model := flag.String("model", "Qwen3.6-35B-A3B-AWQ", "Model name")
	contextSizes := flag.String("contexts", "512,1024,2048,4096,8192", "Comma-separated context sizes to test")
	outputLens := flag.String("outputs", "128,256,512", "Comma-separated output lengths to test")
	concurrency := flag.String("concurrency", "1,2,4,8", "Comma-separated concurrency levels to test")
	numRequests := flag.Int("requests", 10, "Number of requests per scenario")
	apiKey := flag.String("api-key", "", "API key for authentication")
	outputFile := flag.String("output", "", "Output file for JSON results")
	flag.Parse()

	var config Config

	// Load config file if provided
	if *configFile != "" {
		data, err := os.ReadFile(*configFile)
		if err != nil {
			log.Fatalf("Failed to read config file: %v", err)
		}
		if err := json.Unmarshal(data, &config); err != nil {
			log.Fatalf("Failed to parse config file: %v", err)
		}
	} else {
		// Use command line flags
		config.BaseURL = *baseURL
		config.Model = *model
		config.APIKey = *apiKey
		config.NumRequests = *numRequests

		// Parse context sizes
		for _, s := range strings.Split(*contextSizes, ",") {
			var size int
			fmt.Sscanf(strings.TrimSpace(s), "%d", &size)
			config.ContextSizes = append(config.ContextSizes, size)
		}

		// Parse output lengths
		for _, s := range strings.Split(*outputLens, ",") {
			var len int
			fmt.Sscanf(strings.TrimSpace(s), "%d", &len)
			config.OutputLens = append(config.OutputLens, len)
		}

		// Parse concurrency levels
		for _, s := range strings.Split(*concurrency, ",") {
			var c int
			fmt.Sscanf(strings.TrimSpace(s), "%d", &c)
			config.Concurrency = append(config.Concurrency, c)
		}
	}

	fmt.Printf("LMDeploy Benchmark Tool\n")
	fmt.Printf("======================\n")
	fmt.Printf("Server URL: %s\n", config.BaseURL)
	fmt.Printf("Model: %s\n", config.Model)
	fmt.Printf("Context sizes: %v\n", config.ContextSizes)
	fmt.Printf("Output lengths: %v\n", config.OutputLens)
	fmt.Printf("Concurrency levels: %v\n", config.Concurrency)
	fmt.Printf("Requests per scenario: %d\n", config.NumRequests)
	fmt.Printf("\n")

	// Test server connection
	fmt.Print("Testing server connection... ")
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(config.BaseURL + "/v1/models")
	if err != nil {
		log.Fatalf("Failed to connect to server: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Fatalf("Server returned status: %d", resp.StatusCode)
	}
	fmt.Println("OK")

	var allResults []BenchmarkResult
	totalScenarios := len(config.ContextSizes) * len(config.OutputLens) * len(config.Concurrency)
	scenarioNum := 0

	startTime := time.Now()

	// Run all benchmarks
	for _, contextSize := range config.ContextSizes {
		for _, outputLen := range config.OutputLens {
			for _, c := range config.Concurrency {
				scenarioNum++
				fmt.Printf("\n[%d/%d] Testing: context=%d, output=%d, concurrency=%d\n",
					scenarioNum, totalScenarios, contextSize, outputLen, c)

				result := runBenchmark(config, contextSize, outputLen, c)
				allResults = append(allResults, result)

				if result.SuccessCount > 0 {
					fmt.Printf("  Success: %d/%d | Prefill: %.0f t/s | Decode: %.2f t/s\n",
						result.SuccessCount, result.NumRequests,
						result.PrefillTokensPerSec, result.DecodeTokensPerSec)
				} else {
					fmt.Printf("  FAILED: %s\n", result.Error)
				}
			}
		}
	}

	totalDuration := time.Since(startTime)

	// Print summary
	printStats(allResults)

	fmt.Printf("\nTotal benchmark time: %v\n", totalDuration)

	// Save to file if specified
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