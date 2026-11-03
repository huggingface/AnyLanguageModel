@testable import AnyLanguageModel

/// Returns the result of `operation`,
/// or `nil` if it doesn't finish within `seconds`, as when it deadlocks.
///
/// Unlike a test's time limit, this doesn't wait for a deadlocked operation,
/// which can't finish when it waits in a way that ignores cancellation.
/// It leaves the operation and returns.
func outcome<T: Sendable>(
    within seconds: UInt64,
    _ operation: @escaping @Sendable () async throws -> T
) async -> Result<T, any Error>? {
    let resumed = Locked(false)
    let tasks = Locked<[Task<Void, Never>]>([])
    let outcome = await withCheckedContinuation {
        (continuation: CheckedContinuation<Result<T, any Error>?, Never>) in
        @Sendable func finish(_ outcome: Result<T, any Error>?) {
            let isFirst = resumed.withLock { resumed in
                defer { resumed = true }
                return !resumed
            }
            if isFirst { continuation.resume(returning: outcome) }
        }
        let work = Task {
            do {
                finish(.success(try await operation()))
            } catch {
                finish(.failure(error))
            }
        }
        let timeout = Task {
            try? await Task.sleep(nanoseconds: seconds * 1_000_000_000)
            finish(nil)
        }
        tasks.withLock { $0 = [work, timeout] }
    }
    // A deadlocked operation can't finish, so leave it; stop the timer.
    tasks.withLock { $0 }.last?.cancel()
    return outcome
}
