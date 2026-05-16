use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::Mutex;

/// Token bucket rate limiter with refill
#[derive(Debug)]
pub struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(rate: u32, burst: u32) -> Self {
        Self {
            tokens: burst as f64,
            max_tokens: burst as f64,
            refill_rate: rate as f64,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }

    fn try_consume(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Global rate limiter
#[derive(Debug)]
pub struct GlobalRateLimiter {
    bucket: Mutex<TokenBucket>,
    total_requests: AtomicU64,
    rate_limited: AtomicU64,
}

impl GlobalRateLimiter {
    pub fn new(rate_per_second: u32, burst_size: u32) -> Self {
        Self {
            bucket: Mutex::new(TokenBucket::new(rate_per_second, burst_size)),
            total_requests: AtomicU64::new(0),
            rate_limited: AtomicU64::new(0),
        }
    }

    pub async fn try_acquire(&self) -> bool {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        let allowed = self.bucket.lock().await.try_consume();
        if !allowed {
            self.rate_limited.fetch_add(1, Ordering::Relaxed);
        }
        allowed
    }

    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    pub fn rate_limited(&self) -> u64 {
        self.rate_limited.load(Ordering::Relaxed)
    }
}

/// Per-IP rate limiter
#[derive(Debug)]
pub struct PerIpRateLimiter {
    buckets: Mutex<HashMap<IpAddr, (TokenBucket, Instant)>>,
    rate: u32,
    burst: u32,
    max_ips: usize,
    total_requests: AtomicU64,
    rate_limited: AtomicU64,
}

impl PerIpRateLimiter {
    pub fn new(rate_per_second: u32, burst_size: u32, max_ips: usize) -> Self {
        Self {
            buckets: Mutex::new(HashMap::with_capacity(max_ips.min(1024))),
            rate: rate_per_second,
            burst: burst_size,
            max_ips,
            total_requests: AtomicU64::new(0),
            rate_limited: AtomicU64::new(0),
        }
    }

    pub async fn try_acquire(&self, ip: IpAddr) -> bool {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let mut buckets = self.buckets.lock().await;

        // Evict stale entries if over limit
        if buckets.len() >= self.max_ips {
            let stale_cutoff = Instant::now()
                .checked_sub(std::time::Duration::from_secs(300))
                .unwrap_or(Instant::now());
            buckets.retain(|_, (_, last)| *last < stale_cutoff);
        }

        let bucket = buckets
            .entry(ip)
            .or_insert_with(|| (TokenBucket::new(self.rate, self.burst), Instant::now()));
        bucket.1 = Instant::now();

        let allowed = bucket.0.try_consume();
        if !allowed {
            self.rate_limited.fetch_add(1, Ordering::Relaxed);
        }
        allowed
    }

    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    pub fn rate_limited(&self) -> u64 {
        self.rate_limited.load(Ordering::Relaxed)
    }

    pub fn tracked_ips(&self) -> usize {
        self.buckets.try_lock().map(|m| m.len()).unwrap_or(0)
    }
}

/// Rate limiter metrics
#[derive(Debug)]
pub struct RateLimiterMetrics {
    pub global_total_requests: u64,
    pub global_rate_limited: u64,
    pub per_ip_total_requests: u64,
    pub per_ip_rate_limited: u64,
    pub tracked_ips: usize,
}
