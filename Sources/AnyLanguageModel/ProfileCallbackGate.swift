/// Runs a session's profile callbacks and lifecycle actions one at a time.
///
/// A profile stores each callback once, and every request of the session calls it.
/// Overlapping requests would otherwise run the same closure at the same time,
/// and the state that it captures would have a data race.
/// Waiting tasks suspend in first-in, first-out order;
/// none of them blocks a thread.
///
/// An operation can make a request on the same session,
/// such as an `onResponse` callback that sends a follow-up prompt.
/// Waiting for the gate there would wait for the operation itself,
/// so a request from inside an operation runs its callbacks in the operation's scope.
/// Operations in one scope run one at a time, too,
/// and the gate stays held until the last of them finishes,
/// including one in an unstructured task that outlives the operation that started it.
/// A request from a detached task doesn't inherit the scope;
/// if the operation waits for that request, it never finishes.
final class ProfileCallbackGate: Sendable {
    private struct State {
        var isHeld = false
        var waiters: [(id: UInt64, continuation: CheckedContinuation<Void, any Error>)] = []
        var nextWaiterID: UInt64 = 0
        var contentionObservers: [CheckedContinuation<Void, Never>] = []
    }

    /// The scopes of the gates that the current task holds, by gate.
    @TaskLocal private static var scopes: [ObjectIdentifier: ProfileCallbackScope] = [:]

    private let state = Locked(State())

    /// Runs `operation` after the operations that started before it finish.
    ///
    /// Throws `CancellationError` if the task is cancelled while it waits.
    func run<T: Sendable>(_ operation: @Sendable () async throws -> T) async throws -> T {
        if let scope = Self.scopes[ObjectIdentifier(self)], scope.enter() {
            defer { scope.exit() }
            return try await scope.nested.run(operation)
        }
        try await acquire()
        let scope = ProfileCallbackScope(gate: self)
        defer { scope.close() }
        var scopes = Self.scopes
        scopes[ObjectIdentifier(self)] = scope
        return try await Self.$scopes.withValue(scopes) {
            try await operation()
        }
    }

    /// Suspends until a task waits for the gate.
    ///
    /// Tests use this to tell that a second operation is waiting
    /// for the one that holds the gate.
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

    private func acquire() async throws {
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
                    if Task.isCancelled { return .failure(CancellationError()) }
                    state.waiters.append((id, continuation))
                    observers = state.contentionObservers
                    state.contentionObservers = []
                    return nil
                }
                for observer in observers { observer.resume() }
                if let outcome { continuation.resume(with: outcome) }
            }
        } onCancel: {
            let waiter = state.withLock { state -> CheckedContinuation<Void, any Error>? in
                guard let index = state.waiters.firstIndex(where: { $0.id == id }) else { return nil }
                return state.waiters.remove(at: index).continuation
            }
            waiter?.resume(throwing: CancellationError())
        }
    }

    /// Passes the gate to the next waiting task, or opens it.
    fileprivate func release() {
        let next = state.withLock { state -> CheckedContinuation<Void, any Error>? in
            guard !state.waiters.isEmpty else {
                state.isHeld = false
                return nil
            }
            return state.waiters.removeFirst().continuation
        }
        next?.resume()
    }
}

/// One hold of a ``ProfileCallbackGate``,
/// with the operations that run in it.
private final class ProfileCallbackScope: Sendable {
    private struct State {
        var isOpen = true
        var activeOperations = 0
        var isReleased = false

        /// Marks the scope released when its operation and nested operations are done.
        mutating func releaseIfDone() -> Bool {
            guard !isOpen, activeOperations == 0, !isReleased else { return false }
            isReleased = true
            return true
        }
    }

    /// The gate for operations that run in this scope.
    let nested = ProfileCallbackGate()
    private let gate: ProfileCallbackGate
    private let state = Locked(State())

    init(gate: ProfileCallbackGate) {
        self.gate = gate
    }

    /// Starts a nested operation, unless the scope released its gate.
    func enter() -> Bool {
        state.withLock { state in
            guard !state.isReleased else { return false }
            state.activeOperations += 1
            return true
        }
    }

    /// Ends a nested operation.
    func exit() {
        let shouldRelease = state.withLock { state in
            state.activeOperations -= 1
            return state.releaseIfDone()
        }
        if shouldRelease { gate.release() }
    }

    /// Ends the operation that holds the gate.
    func close() {
        let shouldRelease = state.withLock { state in
            state.isOpen = false
            return state.releaseIfDone()
        }
        if shouldRelease { gate.release() }
    }
}
