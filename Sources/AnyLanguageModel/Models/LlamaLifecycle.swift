import Foundation

#if Llama
    import LlamaSwift

    extension LlamaRuntime {
        /// Registers ``LlamaLifecycle/shutDown(deadline:)`` to run when the process exits.
        ///
        /// ggml creates static objects as it needs them,
        /// such as its Metal device and buffer types during a model's first load,
        /// and the C runtime destroys static objects and runs exit handlers
        /// in the reverse order of their creation.
        /// A handler that was registered before the first load
        /// would run after ggml's objects are destroyed,
        /// and freeing a model then would crash.
        /// ggml doesn't report when it creates such an object,
        /// so every load registers the handler again when it ends.
        /// The last registration runs first,
        /// and the handler does nothing after its first run.
        ///
        /// If a registration fails,
        /// an earlier one could run after objects that the models use are destroyed,
        /// so the handler then frees nothing.
        static func registerExitHandler() {
            if atexit({ _ = LlamaLifecycle.shared.shutDown() }) != 0 {
                LlamaLifecycle.shared.exitHandlerRegistrationFailed()
            }
        }
    }

    /// Tracks the llama.cpp work in this process,
    /// and tries to free every loaded model when the process exits.
    ///
    /// ggml's Metal backend frees its device in a C++ static destructor at exit,
    /// and it stops the process with an assertion
    /// if a model or a context still has buffers on the device then.
    /// A generation that is still running at exit can also crash,
    /// on the GPU or the CPU,
    /// because the static objects that it uses are destroyed under it.
    ///
    /// The exit handler is best effort.
    /// It must never make an exit worse than an exit without it,
    /// so it never waits without a limit,
    /// and it never frees a model or a context that running work could use.
    /// It stops new loads and generations,
    /// and it waits up to ``exitDeadline`` for the ones that are running:
    /// a load stops at its next progress report,
    /// and a generation stops before its next token or batch of prompt tokens.
    /// If no work is running then,
    /// it frees the cached session context, the projector, and the model
    /// of every loaded ``LlamaLanguageModel``,
    /// including models that Swift objects still hold.
    /// Otherwise it frees nothing.
    /// Freeing only the idle models wouldn't help,
    /// because the busy model's buffers would still cause the assertion.
    ///
    /// The handler can't help, and the assertion can still happen,
    /// in these cases:
    ///
    /// - An exit during the first load,
    ///   because the handler is registered when a load ends.
    ///   An earlier registration would run after ggml's objects are destroyed.
    /// - An exit from inside a load, a generation, or a cleanup,
    ///   such as from a llama.cpp callback.
    ///   The handler sees that its task or thread is running that work,
    ///   which can't end while the handler waits,
    ///   and returns at once without freeing anything.
    ///   On a thread that runs no task and has no cleanup mark,
    ///   such as a thread that llama.cpp or another library owns,
    ///   the handler can't see this,
    ///   so it waits until the deadline and then frees nothing.
    /// - Work that is still running at the deadline.
    /// - An exit that runs no exit handlers, such as `_exit` or a crash.
    ///
    /// After the handler starts, every load and generation throws
    /// ``LanguageModelSession/GenerationError/assetsUnavailable(_:)``.
    ///
    /// Nothing here calls llama.cpp with a lock held.
    /// llama.cpp logs while it frees a model,
    /// and runs the log callback on the same thread,
    /// so a callback that calls `exit` would run the exit handler on that thread,
    /// and a lock that isn't recursive would then never be released.
    final class LlamaLifecycle: @unchecked Sendable {
        static let shared = LlamaLifecycle()

        /// How long the exit handler waits for running loads and generations.
        ///
        /// A running generation stops before its next token,
        /// or before its next batch of prompt tokens,
        /// so the wait is about one call to llama.cpp,
        /// which takes milliseconds to a few hundred milliseconds for a 4B model.
        /// Three seconds leaves room for a much larger model or a slow CPU,
        /// and still ends a process that can't finish its work
        /// soon enough that the delay looks like a slow quit, not a hang.
        static let exitDeadline: TimeInterval = 3

        /// What ``shutDown(deadline:)`` did.
        enum ShutdownResult: Equatable {
            /// It freed every model.
            case freed
            /// It had already run, so it did nothing.
            case alreadyShutDown
            /// It ran inside an operation, so it freed nothing without waiting.
            case insideOperation
            /// Work was still running at the deadline, so it freed nothing.
            case timedOut
            /// A registration of the exit handler failed, so it freed nothing.
            case registrationFailed
        }

        /// The key in a thread's dictionary for the number of operations that the thread runs.
        private static let depthKey = "AnyLanguageModel.LlamaLifecycle.operationDepth"

        /// Guards the properties below, and signals when no operation is running.
        private let condition = NSCondition()
        private var isShuttingDown = false
        private var registrationFailed = false
        private var activeOperations = 0
        /// The number of active operations in each task.
        ///
        /// A key is used only to compare tasks,
        /// and only while its operation is active,
        /// so its task is still running.
        private var activeTasks: [UnsafeCurrentTask: Int] = [:]
        private var resources: [ObjectIdentifier: LlamaModelResources] = [:]

        /// Whether the process is exiting.
        ///
        /// Generations read this before each token,
        /// which costs one uncontended lock and unlock.
        var isStopping: Bool {
            condition.withLock { isShuttingDown }
        }

        /// The number of loads, generations, and cleanups that are running.
        var activeOperationCount: Int {
            condition.withLock { activeOperations }
        }

        /// The error that work throws after the exit handler starts.
        static var exiting: LanguageModelSession.GenerationError {
            .assetsUnavailable(
                .init(debugDescription: "The process is exiting, so llama.cpp work has stopped.")
            )
        }

        /// Throws ``exiting`` if the exit handler has started.
        func stopIfExiting() throws {
            if isStopping { throw Self.exiting }
        }

        // MARK: Requests

        /// A load or a generation, which the exit handler waits for.
        ///
        /// A request can await other code in the middle of its llama.cpp work,
        /// so an operation belongs to the task that began it, not to a thread.
        /// A request suspends its operation while it waits for other code,
        /// such as a tool call,
        /// so that the exit handler doesn't wait for that code.
        /// Use an operation only in the task that began it.
        final class Operation {
            private let lifecycle: LlamaLifecycle
            private var task: UnsafeCurrentTask?
            private var isActive = true

            fileprivate init(lifecycle: LlamaLifecycle, task: UnsafeCurrentTask?) {
                self.lifecycle = lifecycle
                self.task = task
            }

            /// Ends the operation for now, if it is active.
            func suspend() {
                guard isActive else { return }
                isActive = false
                lifecycle.endOperation(in: task)
            }

            /// Starts the operation again, if it is suspended.
            ///
            /// - Throws: ``LlamaLifecycle/exiting`` if the exit handler has started.
            func resume() throws {
                guard !isActive else { return }
                task = try lifecycle.beginOperation()
                isActive = true
            }

            /// Ends the operation.
            func end() {
                suspend()
            }
        }

        /// Starts an operation in the current task, unless the process is exiting.
        ///
        /// End the operation when its llama.cpp work ends.
        ///
        /// - Throws: ``exiting`` if the exit handler has started.
        func begin() throws -> Operation {
            let task = try beginOperation()
            return Operation(lifecycle: self, task: task)
        }

        private func beginOperation() throws -> UnsafeCurrentTask? {
            // The task outlives the operation, which ends in the same task.
            let task = withUnsafeCurrentTask { $0 }
            try condition.withLock {
                guard !isShuttingDown else { throw Self.exiting }
                activeOperations += 1
                if let task { activeTasks[task, default: 0] += 1 }
            }
            return task
        }

        private func endOperation(in task: UnsafeCurrentTask?) {
            condition.withLock {
                if let task {
                    activeTasks[task, default: 1] -= 1
                    if activeTasks[task] == 0 { activeTasks[task] = nil }
                }
                activeOperations -= 1
                if activeOperations == 0 { condition.broadcast() }
            }
        }

        // MARK: Cleanups

        /// Runs synchronous llama.cpp work, unless the process is exiting.
        func perform<Result>(_ body: () throws -> Result) throws -> Result {
            try condition.withLock {
                guard !isShuttingDown else { throw Self.exiting }
                activeOperations += 1
            }
            return try runCounted(body)
        }

        /// Runs `body` as work that the caller has already counted,
        /// marks the current thread as running it,
        /// and stops counting it when it returns.
        private func runCounted<Result>(_ body: () throws -> Result) rethrows -> Result {
            // The body is synchronous, so it stays on this thread.
            let thread = Thread.current.threadDictionary
            let depth = thread[Self.depthKey] as? Int ?? 0
            thread[Self.depthKey] = depth + 1
            defer {
                thread[Self.depthKey] = depth
                condition.withLock {
                    activeOperations -= 1
                    if activeOperations == 0 { condition.broadcast() }
                }
            }
            return try body()
        }

        /// Whether the current thread or task is running an operation.
        private var isInsideOperation: Bool {
            if (Thread.current.threadDictionary[Self.depthKey] as? Int ?? 0) > 0 {
                return true
            }
            guard let task = withUnsafeCurrentTask(body: { $0 }) else { return false }
            return condition.withLock { activeTasks[task] != nil }
        }

        /// Records that a registration of the exit handler failed.
        func exitHandlerRegistrationFailed() {
            condition.withLock { registrationFailed = true }
        }

        /// Records the resources of a model, so that ``shutDown(deadline:)`` can free them.
        func track(_ model: LlamaModelResources) {
            condition.withLock { resources[ObjectIdentifier(model)] = model }
        }

        /// Frees the resources of a model, unless ``shutDown(deadline:)`` freed them already.
        ///
        /// The cleanup runs even while the process is exiting.
        /// It counts as running work,
        /// so that the exit handler's thread check and deadline apply to it,
        /// and it calls llama.cpp without the lock.
        func free(_ model: LlamaModelResources) {
            condition.withLock {
                resources[ObjectIdentifier(model)] = nil
                activeOperations += 1
            }
            runCounted { model.free() }
        }

        /// Stops new work, waits up to `deadline` for the work that is running,
        /// and frees the resources of every model if no work is running then.
        @discardableResult
        func shutDown(deadline: TimeInterval = exitDeadline) -> ShutdownResult {
            let insideOperation = isInsideOperation
            var models: [LlamaModelResources] = []
            let result: ShutdownResult = condition.withLock {
                guard !isShuttingDown else { return .alreadyShutDown }
                isShuttingDown = true
                // The operation on this thread or task can't end while it waits here.
                guard !insideOperation else { return .insideOperation }
                guard !registrationFailed else { return .registrationFailed }
                // Measure the deadline with a monotonic clock,
                // and wait in short steps that start from the current wall-clock time,
                // so that a change to the wall clock can't extend the wait.
                let clock = ContinuousClock()
                let end = clock.now + .seconds(deadline)
                while activeOperations > 0 {
                    let remaining = clock.now.duration(to: end)
                    guard remaining > .zero else { break }
                    let step = min(remaining, .milliseconds(100))
                    _ = condition.wait(until: Date(timeIntervalSinceNow: step / .seconds(1)))
                }
                guard activeOperations == 0 else { return .timedOut }
                models = Array(resources.values)
                resources = [:]
                return .freed
            }
            // Free without the lock, as in `free(_:)`.
            // If a log callback calls `exit` here, this handler runs again and does nothing.
            for model in models {
                model.free()
            }
            return result
        }
    }

    /// A loaded llama.cpp model, its projector, and its cached session context.
    ///
    /// ``LlamaLifecycle`` holds these resources until it frees them,
    /// separately from the ``LlamaLanguageModel`` that uses them,
    /// so that the exit handler can free a model that a Swift object still holds,
    /// and so that the model's deinit frees nothing twice.
    final class LlamaModelResources: @unchecked Sendable {
        let model: OpaquePointer
        let projector: OpaquePointer?

        /// Guards `cachedContext` and `isFreed`.
        private let lock = NSLock()
        private var cachedContext: OpaquePointer?
        private var isFreed = false

        init(model: OpaquePointer, projector: OpaquePointer?) {
            self.model = model
            self.projector = projector
        }

        /// Records the model's cached session context, or `nil` if it has none.
        ///
        /// Any other context exists only during a load or a generation,
        /// which frees it before it ends.
        func setCachedContext(_ context: OpaquePointer?) {
            lock.withLock { cachedContext = context }
        }

        /// Frees the cached context, the projector, and the model, once.
        ///
        /// Call it only through ``LlamaLifecycle``,
        /// when no load or generation uses the model.
        func free() {
            // Take the context under the lock, and call llama.cpp without it.
            let (isFirst, context): (Bool, OpaquePointer?) = lock.withLock {
                guard !isFreed else { return (false, nil) }
                isFreed = true
                defer { cachedContext = nil }
                return (true, cachedContext)
            }
            guard isFirst else { return }
            if let context {
                llama_free(context)
            }
            if let projector {
                mtmd_free(projector)
            }
            llama_model_free(model)
        }
    }

    #if DEBUG
        /// Functions that tests run at points inside llama.cpp work.
        ///
        /// Only debug builds, which `swift test` uses, compile these hooks,
        /// so they cost nothing in a release build.
        enum LlamaTestHooks {
            /// Runs at the start of each step of a chat generation's token loop,
            /// before its checks for cancellation and exit,
            /// on the generation's thread.
            ///
            /// After the hook returns,
            /// the generation does no llama.cpp work before those checks,
            /// so a generation that the hook holds until the exit handler starts
            /// stops at once.
            static let beforeToken = Locked<(@Sendable () -> Void)?>(nil)

            /// Runs for each llama.cpp log message, before the level filter,
            /// on the thread that logs it.
            static let onLog = Locked<(@Sendable () -> Void)?>(nil)
        }
    #endif
#endif  // Llama
