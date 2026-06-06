# ternary-stream-queue

A CPU-side simulation of **GPU stream semantics** for ternary computation pipelines. Provides FIFO kernel queues, completion events, stream synchronization, multi-stream scheduling, and dependency tracking — all designed to mirror the asynchronous execution model found in CUDA/ROCm but running on the CPU.

## Why?

When building ternary neural network inference engines, you often need GPU-style stream semantics without an actual GPU:

- **Batch scheduling** — Queue kernels (matmul, activate, quantize) in order
- **Multi-stream parallelism** — Run independent pipelines concurrently
- **Event-based sync** — Record events at stream boundaries and wait across streams
- **Dependency tracking** — Enforce ordering between producer/consumer streams

This crate gives you all of that as pure Rust, no GPU required.

## Quick Start

```rust
use ternary_stream_queue::{StreamQueue, FnKernel, StreamEvent, StreamScheduler, StreamSync};

// Simple single-stream FIFO execution
let stream = StreamQueue::new();
stream.enqueue(Box::new(FnKernel::new("matmul", || "C = A×B".into())));
stream.enqueue(Box::new(FnKernel::new("quantize", || "ternary_sign".into())));
let results = stream.execute_all_simple();
assert_eq!(results.len(), 2);
```

## Core Concepts

### Streams

A `StreamQueue` is a FIFO queue of kernels. Kernels execute in the order they were enqueued:

```rust
let stream = StreamQueue::new();
let kid = stream.enqueue(Box::new(FnKernel::new("kernel_a", || {
    "result_a".to_string()
})));
// ... enqueue more kernels ...
let results = stream.execute_all_simple(); // runs all in FIFO order
```

### Events

`StreamEvent` provides completion markers. Record an event at a point in a stream — it signals when all preceding kernels finish:

```rust
let event = stream.record_event();
// ... later ...
event.wait(); // blocks until the event's position is reached
```

### Multi-Stream Scheduling

`StreamScheduler` manages multiple streams and can execute them in parallel:

```rust
let scheduler = StreamScheduler::new();
let s1 = scheduler.create_stream();
let s2 = scheduler.create_stream();

// Enqueue work on both streams...
let all_results = scheduler.execute_parallel(); // threads for each stream
```

### Dependencies

Enqueue a kernel that waits for an event from another stream:

```rust
let event = s1.record_event();
s2.enqueue_with_deps(
    Box::new(FnKernel::new("consumer", || "consumed".into())),
    vec![event.id()],
);
```

### StreamSync Helper

`StreamSync` provides a convenient API for cross-stream synchronization:

```rust
let sync = StreamSync::new(scheduler);
sync.create_dependency(event, &s2, Box::new(FnKernel::new("work", || "done".into())));
```

## API Surface

| Type | Description |
|------|-------------|
| `StreamQueue` | FIFO kernel queue with execution |
| `StreamEvent` | Completion marker (signal/wait) |
| `StreamScheduler` | Multi-stream manager |
| `StreamSync` | Cross-stream synchronization helper |
| `EventRegistry` | Shared event lookup across streams |
| `Kernel` trait | Implement for custom kernel types |
| `FnKernel` | Closure-based kernel implementation |
| `KernelResult` | Execution result with metadata |

## Testing

```bash
cargo test
```

13 tests covering FIFO ordering, event completion, sync blocking, dependency ordering, multi-stream independence, parallel execution, event timeouts, and more.

## License

MIT
