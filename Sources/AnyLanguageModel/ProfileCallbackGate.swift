/// Runs a session's profile callbacks and lifecycle actions one at a time.
///
/// A profile stores each callback once, and every request of the session calls it.
/// Overlapping requests would otherwise run the same closure at the same time,
/// and the state that it captures would have a data race.
/// Waiting tasks suspend in first-in, first-out order;
/// none of them blocks a thread.
///
/// Each operation holds a turn while it runs.
/// An operation can make a request on the same session,
/// such as an `onResponse` callback that sends a follow-up prompt.
/// Waiting for the gate there would wait for the operation itself.
/// So an operation that starts in the task of a running operation
/// doesn't wait for the gate; it runs in a turn nested in that operation's turn,
/// because the task can run only one of them at a time.
///
/// An operation that starts in another task inherits the turn,
/// but it can't tell whether the operation that holds the turn still runs.
/// It runs only while the turn's task waits for a request
/// or for a response stream, which lends the turn,
/// and one at a time with the other operations that the turn's task lends it to.
/// When the turn ends, the operation tries the turn that contains it,
/// and then the gate.
/// So a response stream that an operation consumes runs its callbacks,
/// and a request that the operation starts in an unstructured task
/// and doesn't wait for runs its callbacks later, without overlap.
/// If an operation waits for a task that makes a request in another way,
/// such as with `Task.value`, the request waits for the operation,
/// and the operation never finishes.
/// A request from a detached task doesn't inherit the turn;
/// if the operation waits for that request, it never finishes either.
final class ProfileCallbackGate: Sendable {
    private struct State {
        var isHeld = false
        var waiters: [(id: UInt64, continuation: CheckedContinuation<Void, any Error>)] = []
        var nextWaiterID: UInt64 = 0
        var turnWaiterCount = 0
        var contentionObservers: [CheckedContinuation<Void, Never>] = []
    }

    /// The innermost turn of each gate that the current task holds or inherits.
    @TaskLocal private static var turns: [ObjectIdentifier: ProfileCallbackTurn] = [:]

    private let state = Locked(State())

    /// Runs `operation` when no other operation of the gate runs.
    ///
    /// Throws `CancellationError` if the task is cancelled while it waits.
    func run<T: Sendable>(_ operation: @Sendable () async throws -> T) async throws -> T {
        var turn = Self.turns[ObjectIdentifier(self)]
        if let held = turn, held.isHeldByCurrentTask {
            // The operation that holds the turn waits in this task for this one.
            // Take back the turn from operations in other tasks while this one runs.
            let wasLent = await held.reclaim()
            do {
                let result = try await hold(nestedIn: held, operation)
                if wasLent { _ = held.lendIfHeldByCurrentTask() }
                return result
            } catch {
                if wasLent { _ = held.lendIfHeldByCurrentTask() }
                throw error
            }
        }
        while let inherited = turn {
            if try await inherited.waitForLend() {
                defer { inherited.endLentOperation() }
                return try await hold(nestedIn: inherited, operation)
            }
            turn = inherited.parent
        }
        try await acquire()
        defer { release() }
        return try await hold(nestedIn: nil, operation)
    }

    /// Lends the turns that the current task holds
    /// to operations in other tasks while the task waits.
    ///
    /// Call this before the task waits for a request or a response stream,
    /// and pass the result to ``reclaim(_:)`` when the wait ends.
    static func lendHeldTurns() -> [ProfileCallbackTurn] {
        turns.values.filter { $0.lendIfHeldByCurrentTask() }
    }

    /// Takes back the turns that ``lendHeldTurns()`` lent,
    /// after the operations that run in them finish.
    static func reclaim(_ turns: [ProfileCallbackTurn]) async {
        for turn in turns { _ = await turn.reclaim() }
    }

    /// Suspends until a task waits for the gate or for a turn of it.
    ///
    /// Tests use this to tell that a second operation is waiting
    /// for the one that holds the gate.
    func waitUntilContended() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let isContended = state.withLock { state in
                guard state.waiters.isEmpty, state.turnWaiterCount == 0 else { return true }
                state.contentionObservers.append(continuation)
                return false
            }
            if isContended { continuation.resume() }
        }
    }

    private func hold<T: Sendable>(
        nestedIn parent: ProfileCallbackTurn?,
        _ operation: @Sendable () async throws -> T
    ) async throws -> T {
        let turn = ProfileCallbackTurn(gate: self, parent: parent)
        defer { turn.end() }
        var turns = Self.turns
        turns[ObjectIdentifier(self)] = turn
        return try await Self.$turns.withValue(turns) {
            try await operation()
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

    /// Records that a task waits for a turn of this gate.
    fileprivate func beginTurnWait() {
        let observers = state.withLock { state in
            state.turnWaiterCount += 1
            defer { state.contentionObservers = [] }
            return state.contentionObservers
        }
        for observer in observers { observer.resume() }
    }

    /// Records that a task stopped waiting for a turn of this gate.
    fileprivate func endTurnWait() {
        state.withLock { $0.turnWaiterCount -= 1 }
    }
}

/// One operation's hold on a ``ProfileCallbackGate``.
///
/// The task that runs the operation can lend the turn while it waits,
/// so that operations in other tasks that inherit the turn
/// run one at a time until the task takes the turn back.
final class ProfileCallbackTurn: @unchecked Sendable {
    private struct State {
        var isActive = true
        var isLent = false
        var isRunningLentOperation = false
        var waiters: [(id: UInt64, continuation: CheckedContinuation<Bool, any Error>)] = []
        var nextWaiterID: UInt64 = 0
        var reclaimer: CheckedContinuation<Void, Never>?
    }

    /// The turn that this turn is nested in, if any.
    let parent: ProfileCallbackTurn?
    private let gate: ProfileCallbackGate
    private let state = Locked(State())

    // The task that runs the operation.
    // The turn compares it with the current task only while the turn is active,
    // when that task still runs, so no other task can have its address.
    // It's never used to act on the task.
    private let owner: UnsafeCurrentTask?

    fileprivate init(gate: ProfileCallbackGate, parent: ProfileCallbackTurn?) {
        self.gate = gate
        self.parent = parent
        self.owner = withUnsafeCurrentTask { $0 }
    }

    /// Whether the turn is active and the current task runs its operation.
    var isHeldByCurrentTask: Bool {
        state.withLock { state in state.isActive && isOwnedByCurrentTask }
    }

    private var isOwnedByCurrentTask: Bool {
        guard let owner else { return false }
        return withUnsafeCurrentTask { $0 == owner }
    }

    /// Suspends until the turn's task lends the turn
    /// and no other operation runs in it, and returns `true`;
    /// or until the turn ends, and returns `false`.
    ///
    /// After it returns `true`, call ``endLentOperation()`` when the operation finishes.
    fileprivate func waitForLend() async throws -> Bool {
        let id = state.withLock { state in
            defer { state.nextWaiterID += 1 }
            return state.nextWaiterID
        }
        var isWaiting = false
        defer { if isWaiting { gate.endTurnWait() } }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Bool, any Error>) in
                let outcome: Result<Bool, CancellationError>? = state.withLock { state in
                    guard state.isActive else { return .success(false) }
                    if state.isLent, !state.isRunningLentOperation {
                        state.isRunningLentOperation = true
                        return .success(true)
                    }
                    if Task.isCancelled { return .failure(CancellationError()) }
                    state.waiters.append((id, continuation))
                    return nil
                }
                if let outcome {
                    continuation.resume(with: outcome)
                } else {
                    isWaiting = true
                    gate.beginTurnWait()
                }
            }
        } onCancel: {
            let waiter = state.withLock { state -> CheckedContinuation<Bool, any Error>? in
                guard let index = state.waiters.firstIndex(where: { $0.id == id }) else { return nil }
                return state.waiters.remove(at: index).continuation
            }
            waiter?.resume(throwing: CancellationError())
        }
    }

    /// Ends an operation that ran in the lent turn,
    /// and starts the next waiting one or lets the turn's task take the turn back.
    fileprivate func endLentOperation() {
        enum Next {
            case waiter(CheckedContinuation<Bool, any Error>)
            case reclaimer(CheckedContinuation<Void, Never>)
        }
        let next = state.withLock { state -> Next? in
            state.isRunningLentOperation = false
            if state.isLent, !state.waiters.isEmpty {
                state.isRunningLentOperation = true
                return .waiter(state.waiters.removeFirst().continuation)
            }
            if let reclaimer = state.reclaimer {
                state.reclaimer = nil
                return .reclaimer(reclaimer)
            }
            return nil
        }
        switch next {
        case .waiter(let waiter): waiter.resume(returning: true)
        case .reclaimer(let reclaimer): reclaimer.resume()
        case nil: break
        }
    }

    /// Lends the turn if the current task runs its operation
    /// and the turn isn't lent already.
    fileprivate func lendIfHeldByCurrentTask() -> Bool {
        let (isLent, waiter) = state.withLock { state -> (Bool, CheckedContinuation<Bool, any Error>?) in
            guard state.isActive, !state.isLent, isOwnedByCurrentTask else { return (false, nil) }
            state.isLent = true
            guard !state.isRunningLentOperation, !state.waiters.isEmpty else { return (true, nil) }
            state.isRunningLentOperation = true
            return (true, state.waiters.removeFirst().continuation)
        }
        waiter?.resume(returning: true)
        return isLent
    }

    /// Takes back the turn after the operation that runs in it, if any, finishes.
    ///
    /// Returns whether the turn was lent.
    fileprivate func reclaim() async -> Bool {
        var wasLent = false
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let isDone = state.withLock { state in
                wasLent = state.isLent
                state.isLent = false
                guard state.isRunningLentOperation else { return true }
                state.reclaimer = continuation
                return false
            }
            if isDone { continuation.resume() }
        }
        return wasLent
    }

    /// Ends the turn, and sends the operations that wait for it to the turn that contains it.
    fileprivate func end() {
        let waiters = state.withLock { state in
            state.isActive = false
            state.isLent = false
            defer { state.waiters = [] }
            return state.waiters
        }
        for waiter in waiters { waiter.continuation.resume(returning: false) }
    }
}
