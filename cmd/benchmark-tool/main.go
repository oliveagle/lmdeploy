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
	"sync/atomic"
	"time"
)

// Config 压测配置
type Config struct {
	BaseURL     string   `json:"base_url"`
	Model       string   `json:"model"`
	ContextSizes []int   `json:"context_sizes"`
	OutputLens  []int    `json:"output_lens"`
	Concurrency []int    `json:"concurrency"`
	NumRequests int      `json:"num_requests"`
	APIKey      string   `json:"api_key,omitempty"`
	TimeoutSec  int      `json:"timeout_sec"`
}

// Result 单次请求结果
type Result struct {
	Success        bool
	Duration       time.Duration
	TTFT           time.Duration  // Time To First Token
	PrefillTokens  int
	DecodeTokens   int
	Error          string
}

// BenchmarkStats 基准测试统计
type BenchmarkStats struct {
	Scenario            string
	ContextSize         int
	OutputLen           int
	Concurrency         int
	TotalRequests       int
	SuccessCount        int
	ErrorCount          int
	TotalDuration       time.Duration
	AvgTTFT             time.Duration
	P50TTFT             time.Duration
	P95TTFT             time.Duration
	P99TTFT             time.Duration
	AvgPrefillTPS       float64
	AvgDecodeTPS        float64
	OverallTPS          float64
	RequestsPerSec      float64
	TotalPrefillTokens  int
	TotalDecodeTokens   int
}

// OpenAIRequest OpenAI API 请求
type OpenAIRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
}

// OpenAIMessage 消息
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// generatePrompt 生成指定 token 数量的 prompt
func generatePrompt(targetTokens int) string {
	baseText := "This is a benchmark test for measuring LLM inference performance. " +
		"The quick brown fox jumps over the lazy dog. " +
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit. " +
		"Explain the principles of computer science, artificial intelligence, and machine learning in detail. " +
		"Include discussion about neural networks, deep learning, natural language processing, and large language models. " +
		"Provide examples and use cases for each concept mentioned above. "

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

// sendRequest 发送请求（使用流式 API 测量真实 TTFT）
func sendRequest(client *http.Client, config Config, prompt string, maxTokens, contextSize, outputLen int) Result {
	reqBody := OpenAIRequest{
		Model: config.Model,
		Messages: []OpenAIMessage{
			{Role: "user", Content: prompt},
		},
		MaxTokens:   maxTokens,
		Stream:      true, // 使用流式测量真实 TTFT
		Temperature: 0.7,
	}

	jsonData, _ := json.Marshal(reqBody)

	url := fmt.Sprintf("%s/v1/chat/completions", config.BaseURL)
	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+config.APIKey)
	}

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return Result{Success: false, Error: err.Error()}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return Result{Success: false, Error: fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(body))}
	}

	var ttft time.Duration
	var promptTokens int
	var completionTokens int
	var firstTokenReceived bool

	// 逐行读取 SSE 流
	scanner := newSSEScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		// 第一个 token 到达的时间就是 TTFT
		if !firstTokenReceived {
			ttft = time.Since(start)
			firstTokenReceived = true
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err == nil {
			if chunk.Usage.PromptTokens > 0 {
				promptTokens = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens > 0 {
				completionTokens = chunk.Usage.CompletionTokens
			}
		}
	}

	duration := time.Since(start)

	// 如果 API 没有返回 usage，使用传入的参数
	if promptTokens == 0 {
		promptTokens = contextSize
	}
	if completionTokens == 0 {
		completionTokens = outputLen
	}

	return Result{
		Success:       true,
		Duration:      duration,
		TTFT:          ttft,
		PrefillTokens: promptTokens,
		DecodeTokens:  completionTokens,
	}
}

// sseScanner 简易 SSE 流解析器
type sseScanner struct {
	reader *bufio.Reader
	line   string
	err    error
}

func newSSEScanner(r io.Reader) *sseScanner {
	return &sseScanner{reader: bufio.NewReader(r)}
}

func (s *sseScanner) Text() string {
	return s.line
}

func (s *sseScanner) Scan() bool {
	s.line, s.err = s.reader.ReadString('\n')
	if s.err == io.EOF && s.line != "" {
		s.line = strings.TrimSpace(s.line)
		return true
	}
	if s.err != nil {
		return false
	}
	s.line = strings.TrimSpace(s.line)
	return true
}

func (s *sseScanner) Err() error {
	if s.err == io.EOF {
		return nil
	}
	return s.err
}

// runBenchmark 运行单个场景
func runBenchmark(config Config, contextSize, outputLen, concurrency int) BenchmarkStats {
	prompt := generatePrompt(contextSize)
	timeout := time.Duration(config.TimeoutSec) * time.Second
	client := &http.Client{Timeout: timeout}

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, concurrency)
	results := make(chan Result, config.NumRequests)

	startTime := time.Now()
	var successCount, errorCount int64

	// 启动请求
	for i := 0; i < config.NumRequests; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			semaphore <- struct{}{}        // 获取信号量
			defer func() { <-semaphore }() // 释放信号量

			result := sendRequest(client, config, prompt, outputLen, contextSize, outputLen)
			results <- result

			if result.Success {
				atomic.AddInt64(&successCount, 1)
			} else {
				atomic.AddInt64(&errorCount, 1)
			}
		}()
	}

	// 收集结果
	go func() {
		wg.Wait()
		close(results)
	}()

	var ttfts []time.Duration
	var totalPrefillTokens, totalDecodeTokens int
	var totalDuration time.Duration

	for result := range results {
		if result.Success {
			ttfts = append(ttfts, result.TTFT)
			totalPrefillTokens += result.PrefillTokens
			totalDecodeTokens += result.DecodeTokens
			totalDuration += result.Duration
		}
	}

	endTime := time.Now()

	stats := BenchmarkStats{
		Scenario:      fmt.Sprintf("C%d_O%d", contextSize, outputLen),
		ContextSize:   contextSize,
		OutputLen:     outputLen,
		Concurrency:   concurrency,
		TotalRequests: config.NumRequests,
		SuccessCount:  int(successCount),
		ErrorCount:    int(errorCount),
		TotalDuration: endTime.Sub(startTime),
	}

	if len(ttfts) > 0 {
		sort.Slice(ttfts, func(i, j int) bool {
			return ttfts[i] < ttfts[j]
		})

		// 计算 TTFT 统计
		sum := time.Duration(0)
		for _, t := range ttfts {
			sum += t
		}
		stats.AvgTTFT = sum / time.Duration(len(ttfts))
		stats.P50TTFT = ttfts[len(ttfts)*50/100]
		stats.P95TTFT = ttfts[len(ttfts)*95/100]
		stats.P99TTFT = ttfts[len(ttfts)*99/100]

		// 使用 TTFT 作为 prefill 时间，decode 时间为总时间减去 TTFT
		// 计算 TPS
		if totalDuration > 0 && stats.AvgTTFT > 0 {
			// Prefill TPS = ContextSize / TTFT
			stats.AvgPrefillTPS = float64(contextSize) / stats.AvgTTFT.Seconds()

			// Decode TPS = OutputLen / (总时间 - TTFT)
			decodeTime := totalDuration.Seconds() / float64(len(ttfts)) - stats.AvgTTFT.Seconds()
			if decodeTime > 0 {
				stats.AvgDecodeTPS = float64(outputLen) / decodeTime
			}

			// 统计总的 token 数
			stats.TotalPrefillTokens = contextSize * len(ttfts)
			stats.TotalDecodeTokens = outputLen * len(ttfts)
		}

		stats.OverallTPS = float64(stats.TotalPrefillTokens+stats.TotalDecodeTokens) / stats.TotalDuration.Seconds() * float64(concurrency)
		stats.RequestsPerSec = float64(stats.SuccessCount) / stats.TotalDuration.Seconds()
	}

	return stats
}

// printReport 打印报告
func printReport(allStats []BenchmarkStats) {
	// 按 scenario 分组
	scenarios := make(map[string][]BenchmarkStats)
	for _, s := range allStats {
		scenarios[s.Scenario] = append(scenarios[s.Scenario], s)
	}

	fmt.Println("\n" + strings.Repeat("=", 140))
	fmt.Println("LMDeploy Benchmark Report")
	fmt.Println(strings.Repeat("=", 140))

	// Prefill 性能表
	fmt.Println("\n=== PREFILL PERFORMANCE (tokens/second) ===")
	fmt.Printf("%-15s", "Scenario")
	for _, c := range []int{1, 2, 4, 8, 16} {
		fmt.Printf(" | C=%-8d", c)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 140))

	for _, scenario := range sortedKeys(scenarios) {
		fmt.Printf("%-15s", scenario)
		for _, c := range []int{1, 2, 4, 8, 16} {
			found := false
			for _, s := range scenarios[scenario] {
				if s.Concurrency == c && s.AvgPrefillTPS > 0 {
					fmt.Printf(" | %-10.0f", s.AvgPrefillTPS)
					found = true
					break
				}
			}
			if !found {
				fmt.Printf(" | %-10s", "-")
			}
		}
		fmt.Println()
	}

	// Decode 性能表
	fmt.Println("\n=== DECODE PERFORMANCE (tokens/second) ===")
	fmt.Printf("%-15s", "Scenario")
	for _, c := range []int{1, 2, 4, 8, 16} {
		fmt.Printf(" | C=%-8d", c)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 140))

	for _, scenario := range sortedKeys(scenarios) {
		fmt.Printf("%-15s", scenario)
		for _, c := range []int{1, 2, 4, 8, 16} {
			found := false
			for _, s := range scenarios[scenario] {
				if s.Concurrency == c && s.AvgDecodeTPS > 0 {
					fmt.Printf(" | %-10.2f", s.AvgDecodeTPS)
					found = true
					break
				}
			}
			if !found {
				fmt.Printf(" | %-10s", "-")
			}
		}
		fmt.Println()
	}

	// TTFT 表
	fmt.Println("\n=== TIME TO FIRST TOKEN (milliseconds) ===")
	fmt.Printf("%-15s", "Scenario")
	for _, c := range []int{1, 2, 4, 8, 16} {
		fmt.Printf(" | C=%-8d", c)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 140))

	for _, scenario := range sortedKeys(scenarios) {
		fmt.Printf("%-15s", scenario)
		for _, c := range []int{1, 2, 4, 8, 16} {
			found := false
			for _, s := range scenarios[scenario] {
				if s.Concurrency == c && s.AvgTTFT > 0 {
					fmt.Printf(" | %-10.1f", float64(s.AvgTTFT.Milliseconds()))
					found = true
					break
				}
			}
			if !found {
				fmt.Printf(" | %-10s", "-")
			}
		}
		fmt.Println()
	}

	// 吞吐量表
	fmt.Println("\n=== OVERALL THROUGHPUT (tokens/second) ===")
	fmt.Printf("%-15s", "Scenario")
	for _, c := range []int{1, 2, 4, 8, 16} {
		fmt.Printf(" | C=%-8d", c)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 140))

	for _, scenario := range sortedKeys(scenarios) {
		fmt.Printf("%-15s", scenario)
		for _, c := range []int{1, 2, 4, 8, 16} {
			found := false
			for _, s := range scenarios[scenario] {
				if s.Concurrency == c && s.OverallTPS > 0 {
					fmt.Printf(" | %-10.1f", s.OverallTPS)
					found = true
					break
				}
			}
			if !found {
				fmt.Printf(" | %-10s", "-")
			}
		}
		fmt.Println()
	}
}

func sortedKeys(m map[string][]BenchmarkStats) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func main() {
	configFile := flag.String("config", "", "Configuration file (JSON)")
	baseURL := flag.String("url", "http://localhost:8000", "LMDeploy server URL")
	model := flag.String("model", "Qwen3.6-35B-A3B-AWQ", "Model name")
	contextSizes := flag.String("contexts", "512,1024,2048,4096,8192,16384", "Context sizes (tokens)")
	outputLens := flag.String("outputs", "128,256,512", "Output lengths (tokens)")
	concurrency := flag.String("concurrency", "1,2,4,8,16", "Concurrency levels")
	numRequests := flag.Int("requests", 5, "Requests per scenario")
	apiKey := flag.String("api-key", "", "API key")
	timeout := flag.Int("timeout", 300, "Request timeout (seconds)")
	outputFile := flag.String("output", "benchmark_results.json", "Output JSON file")
	flag.Parse()

	var config Config

	if *configFile != "" {
		data, _ := os.ReadFile(*configFile)
		json.Unmarshal(data, &config)
	} else {
		config.BaseURL = *baseURL
		config.Model = *model
		config.APIKey = *apiKey
		config.NumRequests = *numRequests
		config.TimeoutSec = *timeout

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

	fmt.Printf("=== LMDeploy Benchmark Tool ===\n")
	fmt.Printf("Server: %s\n", config.BaseURL)
	fmt.Printf("Model: %s\n", config.Model)
	fmt.Printf("Context sizes: %v\n", config.ContextSizes)
	fmt.Printf("Output lengths: %v\n", config.OutputLens)
	fmt.Printf("Concurrency: %v\n", config.Concurrency)
	fmt.Printf("Requests per scenario: %d\n", config.NumRequests)

	// 测试连接
	fmt.Print("\nTesting connection... ")
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(config.BaseURL + "/v1/models")
	if err != nil || resp.StatusCode != 200 {
		log.Fatalf("Connection failed: %v", err)
	}
	resp.Body.Close()
	fmt.Println("OK")

	var allStats []BenchmarkStats
	totalScenarios := len(config.ContextSizes) * len(config.OutputLens) * len(config.Concurrency)
	scenarioNum := 0
	startTime := time.Now()

	for _, contextSize := range config.ContextSizes {
		for _, outputLen := range config.OutputLens {
			for _, c := range config.Concurrency {
				scenarioNum++
				fmt.Printf("\n[%d/%d] C=%d O=%d concurrency=%d ... ",
					scenarioNum, totalScenarios, contextSize, outputLen, c)

				stats := runBenchmark(config, contextSize, outputLen, c)
				allStats = append(allStats, stats)

				if stats.SuccessCount > 0 {
					fmt.Printf("OK (Prefill: %.0f t/s, Decode: %.2f t/s, TTFT: %.1fms)\n",
						stats.AvgPrefillTPS, stats.AvgDecodeTPS, float64(stats.AvgTTFT.Milliseconds()))
				} else {
					fmt.Printf("FAILED\n")
				}
			}
		}
	}

	totalDuration := time.Since(startTime)

	printReport(allStats)

	fmt.Printf("\nTotal time: %v\n", totalDuration)

	// 保存结果
	outputData, _ := json.MarshalIndent(allStats, "", "  ")
	os.WriteFile(*outputFile, outputData, 0644)
	fmt.Printf("Results saved to: %s\n", *outputFile)
}