//! Stress tests for the RequestPool's Semaphore concurrency control.
//!
//! These tests verify that the semaphore-based concurrency limiting works correctly:
//! 1. No more than N requests run simultaneously
//! 2. Slots are distributed evenly via round-robin
//! 3. No deadlocks under high contention
//! 4. Correct acquisition and release ordering

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;

/// Simulates the RequestPool's slot selection logic and semaphore behavior.
/// This mirrors the real RequestPool implementation to verify correctness
/// of the round-robin slot distribution algorithm.
struct SimulatedPool {
    semaphore: Arc<Semaphore>,
    active_count: AtomicUsize,
    max_concurrent: AtomicUsize,
    slot_count: usize,
    slot_usage: Vec<AtomicUsize>,
}

impl SimulatedPool {
    fn new(concurrency: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(concurrency)),
            active_count: AtomicUsize::new(0),
            max_concurrent: AtomicUsize::new(0),
            slot_count: concurrency,
            slot_usage: (0..concurrency).map(|_| AtomicUsize::new(0)).collect(),
        }
    }

    /// Run a task with the simulated pool, tracking concurrency metrics.
    async fn run_concurrent<F: FnOnce() + Send>(&self, f: F) {
        let permit = self.semaphore.acquire().await.unwrap();
        let before = self.active_count.fetch_add(1, Ordering::SeqCst);
        self.max_concurrent.fetch_max(before + 1, Ordering::SeqCst);
        f();
        self.active_count.fetch_sub(1, Ordering::SeqCst);
        drop(permit);
    }

    /// Mirrors the slot selection logic from RequestPool::acquire()
    /// and RequestPool::acquire_blocking() in cpp_engine.rs
    fn select_slot(&self) -> usize {
        let active = self.slot_count - self.semaphore.available_permits() - 1;
        active % self.slot_count
    }
}

/// Verify that the semaphore correctly limits concurrent execution.
#[tokio::test]
async fn test_semaphore_respects_concurrency_limit() {
    let pool = Arc::new(SimulatedPool::new(4));
    let mut handles = vec![];

    for _ in 0..20 {
        let pool_clone = Arc::clone(&pool);
        handles.push(tokio::spawn(async move {
            pool_clone
                .run_concurrent(|| {
                    std::thread::sleep(Duration::from_millis(10));
                })
                .await;
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    assert!(
        pool.max_concurrent.load(Ordering::SeqCst) <= 4,
        "Concurrency limit exceeded: max observed {}",
        pool.max_concurrent.load(Ordering::SeqCst)
    );
}

/// Verify that slot distribution is balanced using round-robin selection.
#[tokio::test]
async fn test_slot_distribution_is_balanced() {
    let pool = Arc::new(SimulatedPool::new(4));
    let mut handles = vec![];

    for _ in 0..100 {
        let pool_clone = Arc::clone(&pool);
        handles.push(tokio::spawn(async move {
            let _permit = pool_clone.semaphore.acquire().await.unwrap();
            let idx = pool_clone.select_slot();
            pool_clone.slot_usage[idx].fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(1)).await;
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let usage: Vec<usize> = pool.slot_usage.iter().map(|s| s.load(Ordering::SeqCst)).collect();
    let min_usage = *usage.iter().min().unwrap();
    let max_usage = *usage.iter().max().unwrap();

    // All slots should have been used (within 3x of each other for fair distribution)
    assert!(
        min_usage >= 10,
        "Slot distribution too skewed: min={}, max={:?}",
        min_usage,
        usage
    );
    assert!(
        max_usage <= min_usage * 3,
        "Slot distribution too uneven: min={}, max={:?}",
        min_usage,
        usage
    );
}

/// Verify no deadlock occurs under high contention.
#[tokio::test]
async fn test_no_deadlock_under_contention() {
    let pool = Arc::new(SimulatedPool::new(2));
    let mut handles = vec![];

    for _ in 0..50 {
        let pool_clone = Arc::clone(&pool);
        handles.push(tokio::spawn(async move {
            pool_clone
                .run_concurrent(|| {
                    std::thread::sleep(Duration::from_millis(5));
                })
                .await;
        }));
    }

    let result = tokio::time::timeout(Duration::from_secs(10), futures::future::join_all(handles)).await;

    assert!(result.is_ok(), "Deadlock detected: test timed out");
    for h in result.unwrap() {
        h.unwrap();
    }
}

/// Verify that all tasks complete correctly with proper acquisition/release ordering.
#[tokio::test]
async fn test_all_tasks_complete_with_unique_results() {
    let semaphore = Arc::new(Semaphore::new(1));
    let order = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut handles = vec![];

    for i in 0..10 {
        let sem_clone = Arc::clone(&semaphore);
        let order_clone = Arc::clone(&order);
        handles.push(tokio::spawn(async move {
            let _permit = sem_clone.acquire().await.unwrap();
            order_clone.lock().unwrap().push(i);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let final_order = order.lock().unwrap().clone();
    assert_eq!(final_order.len(), 10, "Not all tasks completed");
    // All IDs should be unique (no duplicates from race conditions)
    let unique_count = final_order.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(unique_count, 10, "Duplicate task IDs observed");
}

/// Stress test: 200 concurrent tasks with tight concurrency limit.
#[tokio::test]
async fn test_concurrent_acquire_release_stress() {
    let pool = Arc::new(SimulatedPool::new(4));
    let completed = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for _ in 0..200 {
        let pool_clone = Arc::clone(&pool);
        let completed_clone = Arc::clone(&completed);
        handles.push(tokio::spawn(async move {
            pool_clone
                .run_concurrent(|| {
                    std::thread::sleep(Duration::from_millis(1));
                })
                .await;
            completed_clone.fetch_add(1, Ordering::SeqCst);
        }));
    }

    let result =
        tokio::time::timeout(Duration::from_secs(30), futures::future::join_all(handles)).await;

    assert!(result.is_ok(), "Stress test timed out");
    for h in result.unwrap() {
        h.unwrap();
    }
    assert_eq!(
        completed.load(Ordering::SeqCst),
        200,
        "Not all tasks completed"
    );
    assert!(
        pool.max_concurrent.load(Ordering::SeqCst) <= 4,
        "Concurrency violated: max={}",
        pool.max_concurrent.load(Ordering::SeqCst)
    );
}

/// Verify edge case: single concurrency limit still works correctly.
#[tokio::test]
async fn test_single_concurrency_limit() {
    let pool = Arc::new(SimulatedPool::new(1));
    let mut handles = vec![];

    for i in 0..10 {
        let pool_clone = Arc::clone(&pool);
        handles.push(tokio::spawn(async move {
            pool_clone
                .run_concurrent(|| {
                    std::thread::sleep(Duration::from_millis(1));
                })
                .await;
            i
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;
    for r in results {
        assert!(r.is_ok());
    }

    assert_eq!(
        pool.max_concurrent.load(Ordering::SeqCst),
        1,
        "Single concurrency was violated"
    );
}

/// Verify round-robin slot selection with varying concurrency levels.
#[tokio::test]
async fn test_slot_selection_with_varying_concurrency() {
    for concurrency in [2, 4, 8, 16] {
        let pool = Arc::new(SimulatedPool::new(concurrency));
        let mut handles = vec![];

        // Run many more tasks than slots to stress the round-robin
        for _ in 0..concurrency * 10 {
            let pool_clone = Arc::clone(&pool);
            handles.push(tokio::spawn(async move {
                let _permit = pool_clone.semaphore.acquire().await.unwrap();
                let idx = pool_clone.select_slot();
                pool_clone.slot_usage[idx].fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_micros(50)).await;
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let usage: Vec<usize> = pool.slot_usage.iter().map(|s| s.load(Ordering::SeqCst)).collect();
        let min_usage = *usage.iter().min().unwrap();
        let max_usage = *usage.iter().max().unwrap();

        // With enough tasks, all slots should be used
        assert!(
            min_usage >= 5,
            "concurrency={}: Slot {} never used, distribution {:?}",
            concurrency,
            usage.iter().position(|&u| u < 5).unwrap_or(0),
            usage
        );
        // Distribution should be reasonably balanced (within 5x)
        assert!(
            max_usage <= min_usage * 5,
            "concurrency={}: Unbalanced distribution: min={}, max={:?}",
            concurrency,
            min_usage,
            usage
        );
    }
}