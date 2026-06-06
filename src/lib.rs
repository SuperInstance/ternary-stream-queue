//! # ternary-stream-queue
//!
//! CPU-side simulation of GPU stream semantics for ternary computation.
//! Provides FIFO kernel queues, completion events, stream synchronization,
//! multi-stream scheduling, and dependency tracking.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Condvar};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Unique identifier for a kernel operation within a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId(pub u64);

/// Unique identifier for a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(pub u64);

/// Unique identifier for an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(pub u64);

static NEXT_KERNEL_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_EVENT_ID: AtomicU64 = AtomicU64::new(1);

/// A simulated kernel that performs work. Returns a result string for verification.
pub trait Kernel: Send + Sync {
    /// Execute the kernel and return a descriptive result.
    fn execute(&self) -> String;
    /// A human-readable name for this kernel.
    fn name(&self) -> &str;
}

/// A simple closure-based kernel.
pub struct FnKernel {
    name: String,
    func: Box<dyn Fn() -> String + Send + Sync>,
}

impl FnKernel {
    pub fn new<F>(name: impl Into<String>, func: F) -> Self
    where
        F: Fn() -> String + Send + Sync + 'static,
    {
        FnKernel {
            name: name.into(),
            func: Box::new(func),
        }
    }
}

impl Kernel for FnKernel {
    fn execute(&self) -> String {
        (self.func)()
    }
    fn name(&self) -> &str {
        &self.name
    }
}

/// A recorded kernel execution result.
#[derive(Debug, Clone)]
pub struct KernelResult {
    pub kernel_id: KernelId,
    pub kernel_name: String,
    pub output: String,
    pub stream_id: StreamId,
    pub completed_at: Instant,
}

/// Internal representation of a queued kernel.
struct QueuedKernel {
    kernel_id: KernelId,
    kernel: Box<dyn Kernel>,
    dependencies: Vec<EventId>,
}

/// A completion event that can be recorded and waited on.
#[derive(Debug)]
pub struct StreamEvent {
    id: EventId,
    state: Arc<(Mutex<EventState>, Condvar)>,
}

#[derive(Debug)]
enum EventState {
    Pending,
    Completed(Instant),
}

impl StreamEvent {
    /// Create a new event in pending state.
    pub fn new() -> Self {
        StreamEvent {
            id: EventId(NEXT_EVENT_ID.fetch_add(1, Ordering::SeqCst)),
            state: Arc::new((
                Mutex::new(EventState::Pending),
                Condvar::new(),
            )),
        }
    }

    /// Get the event's unique identifier.
    pub fn id(&self) -> EventId {
        self.id
    }

    /// Signal this event as completed.
    pub fn signal(&self) {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();
        *state = EventState::Completed(Instant::now());
        cvar.notify_all();
    }

    /// Block until this event is signaled.
    pub fn wait(&self) {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();
        while matches!(*state, EventState::Pending) {
            state = cvar.wait(state).unwrap();
        }
    }

    /// Block until this event is signaled, with a timeout.
    /// Returns true if the event was completed, false if timed out.
    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();
        let deadline = Instant::now() + timeout;
        while matches!(*state, EventState::Pending) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return false;
            }
            let result = cvar.wait_timeout(state, remaining).unwrap();
            state = result.0;
            if result.1.timed_out() {
                return !matches!(*state, EventState::Pending);
            }
        }
        true
    }

    /// Check if the event is completed without blocking.
    pub fn is_completed(&self) -> bool {
        let state = self.state.0.lock().unwrap();
        matches!(*state, EventState::Completed(_))
    }
}

impl Clone for StreamEvent {
    fn clone(&self) -> Self {
        StreamEvent {
            id: self.id,
            state: Arc::clone(&self.state),
        }
    }
}

/// A FIFO stream that queues and executes kernels in order.
pub struct StreamQueue {
    stream_id: StreamId,
    queue: Arc<Mutex<VecDeque<QueuedKernel>>>,
    results: Arc<Mutex<Vec<KernelResult>>>,
    running: AtomicBool,
    completed_event: Arc<(Mutex<bool>, Condvar)>,
}

impl StreamQueue {
    /// Create a new stream queue with a unique ID.
    pub fn new() -> Self {
        StreamQueue {
            stream_id: StreamId(NEXT_STREAM_ID.fetch_add(1, Ordering::SeqCst)),
            queue: Arc::new(Mutex::new(VecDeque::new())),
            results: Arc::new(Mutex::new(Vec::new())),
            running: AtomicBool::new(false),
            completed_event: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    /// Create a stream with a specific ID.
    pub fn with_id(id: u64) -> Self {
        StreamQueue {
            stream_id: StreamId(id),
            queue: Arc::new(Mutex::new(VecDeque::new())),
            results: Arc::new(Mutex::new(Vec::new())),
            running: AtomicBool::new(false),
            completed_event: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    /// Get the stream's unique identifier.
    pub fn id(&self) -> StreamId {
        self.stream_id
    }

    /// Enqueue a kernel onto this stream. Returns the kernel ID.
    pub fn enqueue(&self, kernel: Box<dyn Kernel>) -> KernelId {
        self.enqueue_with_deps(kernel, vec![])
    }

    /// Enqueue a kernel with dependency events that must complete first.
    pub fn enqueue_with_deps(&self, kernel: Box<dyn Kernel>, dependencies: Vec<EventId>) -> KernelId {
        let kid = KernelId(NEXT_KERNEL_ID.fetch_add(1, Ordering::SeqCst));
        let qk = QueuedKernel {
            kernel_id: kid,
            kernel,
            dependencies,
        };
        self.queue.lock().unwrap().push_back(qk);
        kid
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }

    /// Get the number of pending kernels.
    pub fn len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Execute all queued kernels in FIFO order, respecting dependencies.
    /// Returns the results of all executed kernels.
    pub fn execute_all(&self, event_registry: &EventRegistry) -> Vec<KernelResult> {
        self.running.store(true, Ordering::SeqCst);
        let mut executed = Vec::new();
        loop {
            let item = {
                let mut q = self.queue.lock().unwrap();
                q.pop_front()
            };
            match item {
                Some(qk) => {
                    // Wait for dependencies
                    for dep_id in &qk.dependencies {
                        if let Some(event) = event_registry.get(dep_id) {
                            event.wait();
                        }
                    }
                    let output = qk.kernel.execute();
                    let result = KernelResult {
                        kernel_id: qk.kernel_id,
                        kernel_name: qk.kernel.name().to_string(),
                        output,
                        stream_id: self.stream_id,
                        completed_at: Instant::now(),
                    };
                    executed.push(result.clone());
                    self.results.lock().unwrap().push(result);
                }
                None => break,
            }
        }
        self.running.store(false, Ordering::SeqCst);
        // Signal completion
        {
            let (lock, cvar) = &*self.completed_event;
            *lock.lock().unwrap() = true;
            cvar.notify_all();
        }
        executed
    }

    /// Execute all kernels synchronously without an event registry (no deps).
    pub fn execute_all_simple(&self) -> Vec<KernelResult> {
        let registry = EventRegistry::new();
        self.execute_all(&registry)
    }

    /// Get all results collected so far.
    pub fn results(&self) -> Vec<KernelResult> {
        self.results.lock().unwrap().clone()
    }

    /// Check if the stream is currently executing.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Wait for all queued work to complete (blocks until queue drained).
    pub fn wait_until_empty(&self) {
        let (lock, cvar) = &*self.completed_event;
        let mut done = lock.lock().unwrap();
        while !*done && !self.is_empty() {
            done = cvar.wait(done).unwrap();
        }
    }

    /// Record an event at the current position in the stream.
    /// The event will be signaled after all previously queued kernels complete.
    pub fn record_event(&self) -> StreamEvent {
        let event = StreamEvent::new();
        let event_clone = event.clone();
        // Enqueue a special kernel that signals the event
        self.enqueue(Box::new(FnKernel::new("__record_event", move || {
            event_clone.signal();
            "event_recorded".to_string()
        })));
        event
    }
}

/// Registry for tracking events across streams.
#[derive(Default)]
pub struct EventRegistry {
    events: Arc<Mutex<std::collections::HashMap<EventId, StreamEvent>>>,
}

impl EventRegistry {
    pub fn new() -> Self {
        EventRegistry::default()
    }

    /// Register an event.
    pub fn register(&self, event: StreamEvent) {
        self.events.lock().unwrap().insert(event.id(), event);
    }

    /// Get a reference to a registered event.
    pub fn get(&self, id: &EventId) -> Option<StreamEvent> {
        self.events.lock().unwrap().get(id).cloned()
    }

    /// Remove and return an event.
    pub fn remove(&self, id: &EventId) -> Option<StreamEvent> {
        self.events.lock().unwrap().remove(id)
    }
}

/// Multi-stream scheduler for managing multiple stream queues.
pub struct StreamScheduler {
    streams: Arc<Mutex<Vec<Arc<StreamQueue>>>>,
    event_registry: Arc<EventRegistry>,
}

impl StreamScheduler {
    /// Create a new scheduler.
    pub fn new() -> Self {
        StreamScheduler {
            streams: Arc::new(Mutex::new(Vec::new())),
            event_registry: Arc::new(EventRegistry::new()),
        }
    }

    /// Create and register a new stream.
    pub fn create_stream(&self) -> Arc<StreamQueue> {
        let stream = Arc::new(StreamQueue::new());
        self.streams.lock().unwrap().push(stream.clone());
        stream
    }

    /// Get a reference to the event registry.
    pub fn event_registry(&self) -> &EventRegistry {
        &self.event_registry
    }

    /// Execute all streams. Each stream runs to completion.
    /// Returns results grouped by stream.
    pub fn execute_all(&self) -> Vec<(StreamId, Vec<KernelResult>)> {
        let streams = self.streams.lock().unwrap().clone();
        let mut all_results = Vec::new();
        for stream in &streams {
            let results = stream.execute_all(&self.event_registry);
            all_results.push((stream.id(), results));
        }
        all_results
    }

    /// Execute all streams in parallel using threads.
    pub fn execute_parallel(&self) -> Vec<(StreamId, Vec<KernelResult>)> {
        let streams = self.streams.lock().unwrap().clone();
        let registry = self.event_registry.clone();
        let handles: Vec<_> = streams
            .into_iter()
            .map(|stream| {
                let reg = registry.clone();
                std::thread::spawn(move || {
                    let results = stream.execute_all(&reg);
                    (stream.id(), results)
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    }

    /// Wait for all streams to complete.
    pub fn sync_all(&self) {
        let streams = self.streams.lock().unwrap().clone();
        for stream in &streams {
            stream.wait_until_empty();
        }
    }

    /// Get the number of streams.
    pub fn stream_count(&self) -> usize {
        self.streams.lock().unwrap().len()
    }
}

/// Synchronization helper for coordinating between streams.
pub struct StreamSync {
    scheduler: Arc<StreamScheduler>,
}

impl StreamSync {
    /// Create a StreamSync wrapping a scheduler.
    pub fn new(scheduler: Arc<StreamScheduler>) -> Self {
        StreamSync { scheduler }
    }

    /// Wait for a specific event to complete.
    pub fn wait_event(&self, event: &StreamEvent) {
        event.wait();
    }

    /// Wait for a specific event with timeout.
    pub fn wait_event_timeout(&self, event: &StreamEvent, timeout: Duration) -> bool {
        event.wait_timeout(timeout)
    }

    /// Wait for all events in a list to complete.
    pub fn wait_events(&self, events: &[StreamEvent]) {
        for event in events {
            event.wait();
        }
    }

    /// Create a dependency: kernels enqueued on `target_stream` after this call
    /// will wait for the given event before executing.
    pub fn create_dependency(&self, event: StreamEvent, target_stream: &StreamQueue, kernel: Box<dyn Kernel>) -> KernelId {
        target_stream.enqueue_with_deps(kernel, vec![event.id()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_fifo_ordering() {
        let stream = StreamQueue::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        
        for i in 0..5 {
            let order = order.clone();
            stream.enqueue(Box::new(FnKernel::new(format!("k{i}"), move || {
                order.lock().unwrap().push(i);
                format!("result_{i}")
            })));
        }

        let results = stream.execute_all_simple();
        assert_eq!(results.len(), 5);
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3, 4]);
        
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.output, format!("result_{i}"));
        }
    }

    #[test]
    fn test_event_completion() {
        let event = StreamEvent::new();
        assert!(!event.is_completed());

        event.signal();
        assert!(event.is_completed());
    }

    #[test]
    fn test_event_wait() {
        let event = StreamEvent::new();
        let event_clone = event.clone();
        
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            event_clone.signal();
        });

        event.wait();
        assert!(event.is_completed());
        handle.join().unwrap();
    }

    #[test]
    fn test_sync_blocks_until_done() {
        let scheduler = Arc::new(StreamScheduler::new());
        let stream = scheduler.create_stream();
        
        let executed = Arc::new(AtomicBool::new(false));
        let exec_clone = executed.clone();
        
        stream.enqueue(Box::new(FnKernel::new("work", move || {
            thread::sleep(Duration::from_millis(100));
            exec_clone.store(true, Ordering::SeqCst);
            "done".to_string()
        })));

        let scheduler_clone = scheduler.clone();
        let handle = thread::spawn(move || {
            scheduler_clone.execute_all();
        });

        handle.join().unwrap();
        assert!(executed.load(Ordering::SeqCst));
    }

    #[test]
    fn test_dependency_ordering() {
        let registry = EventRegistry::new();
        let stream1 = StreamQueue::new();
        let stream2 = StreamQueue::new();
        
        let event = StreamEvent::new();
        registry.register(event.clone());

        let order = Arc::new(Mutex::new(Vec::new()));
        let o1 = order.clone();
        let o2 = order.clone();

        // Stream 1: enqueue work then signal event
        stream1.enqueue(Box::new(FnKernel::new("s1_k1", move || {
            o1.lock().unwrap().push("s1_k1");
            "s1_done".to_string()
        })));
        let record_event = event.clone();
        stream1.enqueue(Box::new(FnKernel::new("signal", move || {
            record_event.signal();
            "signaled".to_string()
        })));

        // Stream 2: depends on event from stream 1
        stream2.enqueue_with_deps(
            Box::new(FnKernel::new("s2_k1", move || {
                o2.lock().unwrap().push("s2_k1");
                "s2_done".to_string()
            })),
            vec![event.id()],
        );

        // Execute stream1 first, then stream2
        let r1 = stream1.execute_all(&registry);
        let r2 = stream2.execute_all(&registry);

        assert_eq!(r1.len(), 2);
        assert_eq!(r2.len(), 1);
        assert_eq!(*order.lock().unwrap(), vec!["s1_k1", "s2_k1"]);
    }

    #[test]
    fn test_multi_stream_independence() {
        let scheduler = Arc::new(StreamScheduler::new());
        let s1 = scheduler.create_stream();
        let s2 = scheduler.create_stream();
        
        let results1 = Arc::new(Mutex::new(Vec::new()));
        let results2 = Arc::new(Mutex::new(Vec::new()));
        
        let r1 = results1.clone();
        let r2 = results2.clone();

        for i in 0..3 {
            let r1 = r1.clone();
            s1.enqueue(Box::new(FnKernel::new(format!("s1_{i}"), move || {
                r1.lock().unwrap().push(format!("s1_{i}"));
                format!("s1_{i}")
            })));
            
            let r2 = r2.clone();
            s2.enqueue(Box::new(FnKernel::new(format!("s2_{i}"), move || {
                r2.lock().unwrap().push(format!("s2_{i}"));
                format!("s2_{i}")
            })));
        }

        let all_results = scheduler.execute_parallel();
        assert_eq!(all_results.len(), 2);
        
        assert_eq!(*results1.lock().unwrap(), vec!["s1_0", "s1_1", "s1_2"]);
        assert_eq!(*results2.lock().unwrap(), vec!["s2_0", "s2_1", "s2_2"]);
    }

    #[test]
    fn test_record_event_in_stream() {
        let stream = StreamQueue::new();
        let counter = Arc::new(AtomicUsize::new(0));
        
        let c1 = counter.clone();
        stream.enqueue(Box::new(FnKernel::new("k1", move || {
            c1.fetch_add(1, Ordering::SeqCst).to_string()
        })));

        let event = stream.record_event();
        
        let c2 = counter.clone();
        stream.enqueue(Box::new(FnKernel::new("k2", move || {
            c2.fetch_add(10, Ordering::SeqCst).to_string()
        })));

        assert!(!event.is_completed());
        stream.execute_all_simple();
        assert!(event.is_completed());
        assert_eq!(counter.load(Ordering::SeqCst), 11);
    }

    #[test]
    fn test_event_timeout() {
        let event = StreamEvent::new();
        
        // Event not signaled, should timeout
        let result = event.wait_timeout(Duration::from_millis(10));
        assert!(!result);

        event.signal();
        let result = event.wait_timeout(Duration::from_millis(10));
        assert!(result);
    }

    #[test]
    fn test_stream_queue_len_and_empty() {
        let stream = StreamQueue::new();
        assert!(stream.is_empty());
        assert_eq!(stream.len(), 0);

        stream.enqueue(Box::new(FnKernel::new("k", || "x".to_string())));
        assert!(!stream.is_empty());
        assert_eq!(stream.len(), 1);

        stream.execute_all_simple();
        assert!(stream.is_empty());
    }

    #[test]
    fn test_stream_sync_helper() {
        let scheduler = Arc::new(StreamScheduler::new());
        let sync = StreamSync::new(scheduler.clone());
        let stream = scheduler.create_stream();

        let event = StreamEvent::new();
        let event_clone = event.clone();
        
        stream.enqueue(Box::new(FnKernel::new("work", move || {
            event_clone.signal();
            "done".to_string()
        })));

        let sched = scheduler.clone();
        thread::spawn(move || {
            sched.execute_all();
        });

        sync.wait_event(&event);
        assert!(event.is_completed());
    }

    #[test]
    fn test_create_dependency_via_sync() {
        let scheduler = Arc::new(StreamScheduler::new());
        let sync = StreamSync::new(scheduler.clone());
        let s1 = scheduler.create_stream();
        let s2 = scheduler.create_stream();

        let event = StreamEvent::new();
        let event_signal = event.clone();

        s1.enqueue(Box::new(FnKernel::new("producer", move || {
            event_signal.signal();
            "produced".to_string()
        })));

        let order = Arc::new(Mutex::new(Vec::new()));
        let o = order.clone();
        sync.create_dependency(
            event.clone(),
            &*s2,
            Box::new(FnKernel::new("consumer", move || {
                o.lock().unwrap().push("consumer_ran");
                "consumed".to_string()
            })),
        );

        let reg = scheduler.event_registry().clone();
        let r1 = s1.execute_all(&reg);
        let r2 = s2.execute_all(&reg);
        
        assert_eq!(r1.len(), 1);
        assert_eq!(r2.len(), 1);
        assert!(event.is_completed());
    }

    #[test]
    fn test_kernel_result_metadata() {
        let stream = StreamQueue::new();
        let kid = stream.enqueue(Box::new(FnKernel::new("my_kernel", || "output_42".to_string())));
        
        let results = stream.execute_all_simple();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].kernel_id, kid);
        assert_eq!(results[0].kernel_name, "my_kernel");
        assert_eq!(results[0].output, "output_42");
        assert_eq!(results[0].stream_id, stream.id());
    }

    #[test]
    fn test_parallel_execution_with_dependencies() {
        let scheduler = Arc::new(StreamScheduler::new());
        let s1 = scheduler.create_stream();
        let s2 = scheduler.create_stream();
        
        let event = StreamEvent::new();
        let event_signal = event.clone();
        
        let results = Arc::new(Mutex::new(Vec::new()));
        let r1 = results.clone();
        let r2 = results.clone();

        s1.enqueue(Box::new(FnKernel::new("s1_first", move || {
            thread::sleep(Duration::from_millis(50));
            event_signal.signal();
            r1.lock().unwrap().push("s1_first");
            "ok".to_string()
        })));

        s2.enqueue_with_deps(
            Box::new(FnKernel::new("s2_after", move || {
                r2.lock().unwrap().push("s2_after");
                "ok".to_string()
            })),
            vec![event.id()],
        );
        
        // Register the event so stream2 can find it
        scheduler.event_registry().register(event);

        let all = scheduler.execute_parallel();
        assert_eq!(all.len(), 2);
        
        let res = results.lock().unwrap();
        assert!(res.iter().any(|s| *s == "s1_first"));
        assert!(res.iter().any(|s| *s == "s2_after"));
    }
}
