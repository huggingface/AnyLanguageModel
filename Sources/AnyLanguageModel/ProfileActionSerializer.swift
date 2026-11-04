/// Runs a session's profile callbacks and lifecycle actions one at a time.
///
/// A profile stores each callback once, and every request of the session calls it.
/// Overlapping requests would otherwise run the same closure at the same time,
/// and the state that it captures would have a data race.
///
/// The session has one permit.
/// An operation runs only while it holds the permit,
/// and tasks that wait for the permit suspend in first-in, first-out order;
/// none of them blocks a thread.
///
/// An operation can make a request on any session,
/// such as an `onResponse` callback that sends a follow-up prompt.
/// While the operation's task waits for a request or for a response stream
/// (in ``withPermitsReleased(isolation:_:)``),
/// it releases the permits that it holds,
/// so the request's own operations and the operations of other requests can run.
/// Before the task continues, it takes back each permit,
/// after the operations that run in the meantime end or release it in turn.
/// So, as with actor reentrancy, other operations can run
/// while an operation waits for a request,
/// but never at the same time as the operation's own code.
///
/// An operation that starts in the task that holds the permit,
/// outside such a wait, runs without waiting.
/// An operation in another task, such as a child task or an unstructured task,
/// waits for the permit.
/// So a request that a callback starts in an unstructured task and doesn't wait for
/// runs its callbacks after the callback returns or while it waits for a request.
/// If an operation waits for another task that runs an operation of the same session,
/// that task waits for the permit, and the operation never finishes.
/// This includes a child task, from `async let` or a task group,
/// and an unstructured or detached task that the operation awaits with `Task.value`.
///
/// Lifecycle actions also wait for the session's previous preparation,
/// which the permit doesn't cover;
/// ``ProfileLifecycle/Preparation`` detects cycles of those waits.
final class ProfileActionSerializer: Sendable {
    private struct State {
        var isHeld = false
        var waiters: [(id: UInt64, continuation: CheckedContinuation<Void, any Error>)] = []
        var nextWaiterID: UInt64 = 0
        var contentionObservers: [CheckedContinuation<Void, Never>] = []
    }

    /// The holders of the operations that the current task runs,
    /// or that the tasks that it inherits from run, innermost last.
    @TaskLocal private static var holders: [Holder] = []

    private let state = Locked(State())

    /// Runs `operation` while it holds the permit.
    ///
    /// Throws `CancellationError` if the task is cancelled while it waits for the permit.
    func run<T: Sendable>(_ operation: @Sendable () async throws -> T) async throws -> T {
        if let holder = Self.holders.last(where: { $0.serializer === self }), holder.isHeldByCurrentTask {
            // The operation that holds the permit waits in this task for this one.
            return try await operation()
        }
        try await acquire(isCancellable: true)
        let holder = Holder(serializer: self)
        defer {
            holder.end()
            release()
        }
        return try await Self.$holders.withValue(Self.holders.filter(\.isActive) + [holder]) {
            try await operation()
        }
    }

    /// Releases the permits that the current task holds while `body` runs,
    /// and takes them back before it returns or throws.
    ///
    /// Call this where the task waits for a request or a response stream,
    /// so that the operations that the request runs don't wait for the task.
    /// Taking a permit back ignores cancellation,
    /// because the operation that holds it continues after this returns.
    static func withPermitsReleased<T>(
        isolation: isolated (any Actor)? = #isolation,
        _ body: () async throws -> T
    ) async throws -> T {
        let released = holders.filter { $0.releaseIfHeldByCurrentTask() }
        guard !released.isEmpty else { return try await body() }
        do {
            let result = try await body()
            for holder in released.reversed() { await holder.reacquire() }
            return result
        } catch {
            for holder in released.reversed() { await holder.reacquire() }
            throw error
        }
    }

    /// Suspends until a task waits for the permit.
    ///
    /// Tests use this to tell that a second operation is waiting
    /// for the one that holds the permit.
    func waitUntilContended() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let isContended = state.withLock { state in
                guard state.waiters.isEmpty else { return true }
                state.contentionObservers.append(continuation)
                return false
            }
            if isContended { continuation.resume() }
        }
    }

    private func acquire(isCancellable: Bool) async throws {
        let id = state.withLock { state in
            defer { state.nextWaiterID += 1 }
            return state.nextWaiterID
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                var observers: [CheckedContinuation<Void, Never>] = []
                let outcome: Result<Void, CancellationError>? = state.withLock { state in
                    if !state.isHeld {
                        state.isHeld = true
                        return .success(())
                    }
                    if isCancellable, Task.isCancelled { return .failure(CancellationError()) }
                    state.waiters.append((id, continuation))
                    observers = state.contentionObservers
                    state.contentionObservers = []
                    return nil
                }
                for observer in observers { observer.resume() }
                if let outcome { continuation.resume(with: outcome) }
            }
        } onCancel: {
            guard isCancellable else { return }
            let waiter = state.withLock { state -> CheckedContinuation<Void, any Error>? in
                guard let index = state.waiters.firstIndex(where: { $0.id == id }) else { return nil }
                return state.waiters.remove(at: index).continuation
            }
            waiter?.resume(throwing: CancellationError())
        }
    }

    /// Passes the permit to the next waiting task, or makes it available.
    private func release() {
        let next = state.withLock { state -> CheckedContinuation<Void, any Error>? in
            guard !state.waiters.isEmpty else {
                state.isHeld = false
                return nil
            }
            return state.waiters.removeFirst().continuation
        }
        next?.resume()
    }

    /// One operation's claim on the permit.
    ///
    /// The holder records the task that runs the operation,
    /// so that only that task runs nested operations without waiting
    /// and releases the permit while it waits.
    private final class Holder: @unchecked Sendable {
        private struct State {
            var isActive = true
            var hasPermit = true
        }

        let serializer: ProfileActionSerializer
        private let state = Locked(State())

        // The hash value of the task that runs the operation.
        // An `UnsafeCurrentTask` is valid only in the closure that receives it,
        // so the holder keeps this value instead of the task,
        // and it computes the current task's hash value in the same way to compare them.
        // The holder compares them only while it is active,
        // when the operation's task still runs, so no other task can have its address;
        // two tasks that run at the same time have equal hash values
        // only if the hashes of their different addresses collide.
        private let owner: Int?

        init(serializer: ProfileActionSerializer) {
            self.serializer = serializer
            self.owner = Self.currentTaskHashValue
        }

        private static var currentTaskHashValue: Int? {
            withUnsafeCurrentTask { $0?.hashValue }
        }

        /// Whether the operation still runs.
        var isActive: Bool { state.withLock { $0.isActive } }

        /// Whether the current task runs the operation and holds the permit for it.
        var isHeldByCurrentTask: Bool {
            state.withLock { state in state.isActive && state.hasPermit && isOwnedByCurrentTask }
        }

        private var isOwnedByCurrentTask: Bool {
            guard let owner else { return false }
            return Self.currentTaskHashValue == owner
        }

        /// Releases the permit if the current task holds it for the operation.
        func releaseIfHeldByCurrentTask() -> Bool {
            let isReleased = state.withLock { state in
                guard state.isActive, state.hasPermit, isOwnedByCurrentTask else { return false }
                state.hasPermit = false
                return true
            }
            if isReleased { serializer.release() }
            return isReleased
        }

        /// Takes back the permit that ``releaseIfHeldByCurrentTask()`` released.
        func reacquire() async {
            // A wait that ignores cancellation doesn't throw.
            try? await serializer.acquire(isCancellable: false)
            state.withLock { $0.hasPermit = true }
        }

        /// Records that the operation ended.
        func end() {
            state.withLock { $0.isActive = false }
        }
    }
}
