import Dispatch

/// A task executor that runs jobs on its own serial dispatch queue,
/// outside the cooperative thread pool of Swift concurrency.
///
/// Tests that pause synchronous work by blocking a thread,
/// such as a profile body that waits on a semaphore,
/// run that work in a task that prefers this executor.
/// Blocking a cooperative thread instead can use up the pool
/// when tests run in parallel on a machine with few cores,
/// and then the task that would unblock the work never runs.
@available(macOS 15.0, iOS 18.0, macCatalyst 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
final class BlockingTaskExecutor: TaskExecutor {
    private let queue = DispatchQueue(label: "AnyLanguageModelTests.BlockingTaskExecutor")

    func enqueue(_ job: consuming ExecutorJob) {
        let job = UnownedJob(job)
        queue.async {
            job.runSynchronously(on: self.asUnownedTaskExecutor())
        }
    }
}
