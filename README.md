# ternary-stream-queue

GPU stream semantics on the CPU — FIFO kernel queues, completion events, multi-stream scheduling, and dependency tracking for ternary computation pipelines.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Why this exists

GPUs execute kernels asynchronously on ordered streams. You enqueue a matmul, then a quantize, then a copy — they run in FIFO order without blocking the host. When you need synchronization, you record an event and wait on it from another stream. This is the CUDA/ROCm execution model, and it's essential for overlapping computation with communication.

This crate brings that model to the CPU for ternary inference pipelines. You get the same programming pattern — enqueue kernels, record events, create cross-stream dependencies — without needing a GPU. It's a testing, prototyping, and educational tool that lets you design your stream schedule before committing to hardware.

## The key insight

The GPU stream model solves a fundamental problem: *how do you express "A must finish before B starts" without threads, mutexes, or explicit barriers?* The answer: events. You record an event at position A, then B waits on that event. The stream runtime handles the scheduling. This crate implements the same abstraction with `std::sync::Condvar` — events are signaled and waited on using the same semantics as CUDA events, but running on POSIX threads.

## Quick Start

```rust
use ternary_stream_queue::{StreamQueue, FnKernel, StreamEvent, StreamScheduler, StreamSync};

// ── Single stream: FIFO execution ──
let stream = StreamQueue::new();
stream.enqueue(Box::new(FnKernel::new("matmul", || "C = A×B".into())));
stream.enqueue(Box::new(FnKernel::new("quantize", || "ternary_sign".into())));
let results = stream.execute_all_simple();
assert_eq!(results.len(), 2);

// ── Events: record and wait ──
let event = stream.record_event(); // signals after all preceding kernels finish
// ... later ...
event.wait(); // blocks until signaled

// ── Multi-stream parallel execution ──
let scheduler = StreamScheduler::new();
let s1 = scheduler.create_stream();
let s2 = scheduler.create_stream();
// Enqueue work on both streams...
let all_results = scheduler.execute_parallel(); // each stream on its own thread

// ── Cross-stream dependencies ──
let event = s1.record_event();
s2.enqueue_with_deps(
    Box::new(FnKernel::new("consumer", || "consumed".into())),
    vec![event.id()],
);

// ── StreamSync helper ──
let sync = StreamSync::new(Arc::new(scheduler));
sync.create_dependency(event, &s2, Box::new(FnKernel::new("work", || "done".into())));
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    StreamScheduler                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Stream 0 │  │ Stream 1 │  │ Stream 2 │          │
│  │ [K0,K1,K2]│  │ [K3,K4]  │  │ [K5,K6,K7]│        │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘          │
│        │              │              │               │
│        ▼              ▼              ▼               │
│  execute_parallel() ──→ one thread per stream        │
│        │              │              │               │
│   K0 ─→ K1 ─→ K2    K3 ─→ K4    K5 ─→ K6 ─→ K7   │
│                  ▲                        ▲          │
│                  │     EventRegistry       │          │
│                  └──── event from K2 ──────┘          │
│                   (K6 waits for K2 to finish)        │
└─────────────────────────────────────────────────────┘
```

Each stream is a `StreamQueue` (FIFO, thread-safe). Events are condition-variable-backed completion markers. The `StreamScheduler` manages multiple streams and can execute them in parallel (one thread per stream) or sequentially. Dependencies are expressed as event IDs — a kernel with dependencies blocks until all its events are signaled.

## API Reference

### Core Types

```rust
pub trait Kernel: Send + Sync {
    fn execute(&self) -> String;
    fn name(&self) -> &str;
}

pub struct FnKernel { /* closure-based kernel */ }
FnKernel::new(name, || -> String)

pub struct KernelResult {
    pub kernel_id: KernelId,
    pub kernel_name: String,
    pub output: String,
    pub stream_id: StreamId,
    pub completed_at: Instant,
}
```

### StreamQueue

```rust
let stream = StreamQueue::new();
stream.enqueue(kernel: Box<dyn Kernel>) -> KernelId
stream.enqueue_with_deps(kernel, dependencies: Vec<EventId>) -> KernelId
stream.record_event() -> StreamEvent
stream.execute_all(registry: &EventRegistry) -> Vec<KernelResult>
stream.execute_all_simple() -> Vec<KernelResult>  // no deps
stream.wait_until_empty()
stream.len() -> usize
stream.is_empty() -> bool
stream.is_running() -> bool
```

### StreamEvent

```rust
let event = StreamEvent::new();
event.signal()                // mark as completed
event.wait()                  // block until signaled
event.wait_timeout(duration) -> bool  // timed wait
event.is_completed() -> bool  // non-blocking check
```

### StreamScheduler

```rust
let scheduler = StreamScheduler::new();
scheduler.create_stream() -> Arc<StreamQueue>
scheduler.execute_all() -> Vec<(StreamId, Vec<KernelResult>)>      // sequential
scheduler.execute_parallel() -> Vec<(StreamId, Vec<KernelResult>)> // threaded
scheduler.sync_all()
scheduler.event_registry() -> &EventRegistry
```

### StreamSync

```rust
let sync = StreamSync::new(Arc::new(scheduler));
sync.wait_event(event)
sync.wait_event_timeout(event, timeout) -> bool
sync.wait_events(events)
sync.create_dependency(event, target_stream, kernel) -> KernelId
```

### EventRegistry

```rust
let registry = EventRegistry::new();
registry.register(event)
registry.get(&event_id) -> Option<StreamEvent>
registry.remove(&event_id) -> Option<StreamEvent>
```

## Real-world example

A fishing boat runs a ternary inference pipeline with two accelerators. Stream 0 processes sonar data (matmul → quantize → classify). Stream 1 processes camera data (conv → quantize → classify). Both streams feed into Stream 2, which fuses the results.

```
Stream 0: [sonar_matmul] → [quantize] → [classify] ──→ event_sonar_done
Stream 1: [camera_conv] → [quantize] → [classify] ──→ event_camera_done
Stream 2: [fuse(deps: event_sonar, event_camera)] → [decision]
```

Stream 2's fuse kernel has dependencies on both events. It won't execute until both classification results are ready. If the sonar pipeline is slower (more tokens), the camera pipeline finishes early and Stream 2's thread blocks efficiently on the sonar event — no busy-waiting.

## Ecosystem connections

- **[`ternary-transformer`](https://github.com/SuperInstance/ternary-transformer)** — the model whose forward/backward passes are scheduled on streams
- **[`ternary-command-buffer`](https://github.com/SuperInstance/ternary-command-buffer)** — records operations for replay; streams execute them
- **[`ternary-pipeline-parallel`](https://github.com/SuperInstance/ternary-pipeline-parallel)** — pipeline stages map naturally to streams
- **[`ternary-tensor-parallel`](https://github.com/SuperInstance/ternary-tensor-parallel)** — tensor-parallel devices use independent streams with cross-stream sync

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `enqueue` | O(1) | Lock + push back |
| `execute_all` | O(k) per stream | k = queued kernels |
| `execute_parallel` | O(k/p) wall time | p = parallel streams, assumes balanced work |
| `event.wait` | O(1) wake | Condvar signal/broadcast |
| `event.wait_timeout` | O(timeout) worst | Returns false if timed out |

Thread overhead: one thread per stream during `execute_parallel`. For CPU-bound ternary kernels, the threading overhead is negligible compared to the compute.

## Open questions

- **Priority queues**: Right now all kernels in a stream have equal priority. For latency-sensitive inference, some kernels (the final classification) should preempt others (background logging).
- **Stream pools**: Instead of one thread per stream, a thread pool with work-stealing would handle imbalanced workloads better.
- **Backpressure**: If Stream 0 enqueues faster than Stream 2 can consume, the queue grows unbounded. Should there be a bounded queue with backpressure?
- **GPU backend**: The API mirrors CUDA streams closely enough that a GPU backend (via CUDA Driver API) could implement the same `Kernel` trait with actual GPU kernel launches.

## Testing

```bash
cargo test
```

13 tests: FIFO ordering (5 kernels execute in sequence), event signal/wait, cross-thread event synchronization, dependency ordering (consumer waits for producer), multi-stream independence, parallel execution correctness, recorded events in streams, event timeouts (expired and non-expired), queue length and emptiness, StreamSync helper, metadata on KernelResult, parallel execution with cross-stream dependencies.

## License

MIT
