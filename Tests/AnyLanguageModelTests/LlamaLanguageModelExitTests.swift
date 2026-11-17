import Foundation
import Testing

@testable import AnyLanguageModel

#if Llama
    /// A model that a global variable holds for the life of the process.
    nonisolated(unsafe) private var globalExitTestModel: LlamaLanguageModel?

    /// Tests that exit a child process with a model loaded.
    ///
    /// The tests that use ``LlamaTestHooks`` run only in debug builds,
    /// which compile the hooks.
    ///
    /// Without the exit handler,
    /// llama.cpp's Metal backend stops such a process with an assertion
    /// when the model has layers on the GPU,
    /// and an exit during a generation crashes on the GPU and on the CPU.
    @Suite(
        "LlamaLanguageModel at exit",
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] != nil)
    )
    struct LlamaLanguageModelExitTests {
        static var modelPath: String {
            ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] ?? ""
        }

        /// Returns a model and a session after a first response,
        /// so that the model and its cached context are loaded.
        static func loadedModel(gpuLayers: Int32) async throws -> (LlamaLanguageModel, LanguageModelSession) {
            let model = LlamaLanguageModel(modelPath: modelPath, gpuLayers: gpuLayers)
            let session = LanguageModelSession(model: model)
            _ = try await session.respond(to: "Say hello", options: GenerationOptions(maximumResponseTokens: 2))
            return (model, session)
        }

        #if DEBUG
            /// Holds a generation until the exit handler has started,
            /// so that the handler must wait for it.
            static func holdUntilShutdownStarts() {
                let limit = ExitTestTimer.now() + 10_000_000_000
                while !LlamaLifecycle.shared.isStopping {
                    // A distinct status shows that the exit handler never started.
                    if ExitTestTimer.now() > limit { _exit(9) }
                    usleep(1_000)
                }
            }

            /// Starts a generation in another task,
            /// and returns when its first token step starts.
            ///
            /// - Parameter whileActive: What the generation does at the start of that step,
            ///   while it is known to be running.
            static func startGeneration(
                _ session: LanguageModelSession,
                whileActive: @escaping @Sendable () -> Void
            ) {
                let started = DispatchSemaphore(value: 0)
                let once = Locked(false)
                LlamaTestHooks.beforeToken.withLock {
                    $0 = {
                        let isFirst = once.withLock { done in
                            defer { done = true }
                            return !done
                        }
                        guard isFirst else { return }
                        started.signal()
                        whileActive()
                    }
                }
                nonisolated(unsafe) let session = session
                Task.detached {
                    _ = try? await session.respond(
                        to: "Count from 1 to 1000.",
                        options: GenerationOptions(maximumResponseTokens: 2048)
                    )
                }
                // A distinct status shows that the generation never started.
                if started.wait(timeout: .now() + 30) == .timedOut { _exit(3) }
            }
        #endif

        @Test("A process exits normally with a loaded model", arguments: [Int32(-1), 0])
        func idleExit(gpuLayers: Int32) async throws {
            let name = "idleExit\(gpuLayers)"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) { [name = name as String, gpuLayers = gpuLayers as Int32] in
                ExitTestTimer.childStarted(name)
                let (model, session) = try await LlamaLanguageModelExitTests.loadedModel(gpuLayers: gpuLayers)
                ExitTestTimer.stepStarted(name)
                withExtendedLifetime((model, session)) { exit(0) }
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }

        @Test("A process exits normally with a model in a global variable", arguments: [Int32(-1), 0])
        func exitWithGlobalModel(gpuLayers: Int32) async throws {
            let name = "exitWithGlobalModel\(gpuLayers)"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) { [name = name as String, gpuLayers = gpuLayers as Int32] in
                ExitTestTimer.childStarted(name)
                globalExitTestModel = LlamaLanguageModel(
                    modelPath: LlamaLanguageModelExitTests.modelPath,
                    gpuLayers: gpuLayers
                )
                _ = try await LanguageModelSession(model: globalExitTestModel!).respond(
                    to: "Say hello",
                    options: GenerationOptions(maximumResponseTokens: 2)
                )
                ExitTestTimer.stepStarted(name)
                exit(0)
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }

        #if DEBUG
            @Test(
                "The exit handler waits for a generation that is running, and the process exits normally",
                arguments: [Int32(-1), 0]
            )
            func exitDuringGeneration(gpuLayers: Int32) async throws {
                let name = "exitDuringGeneration\(gpuLayers)"
                let watchdog = ExitTestTimer.watch(name)
                await #expect(processExitsWith: .success) { [name = name as String, gpuLayers = gpuLayers as Int32] in
                    ExitTestTimer.childStarted(name)
                    let (model, session) = try await LlamaLanguageModelExitTests.loadedModel(gpuLayers: gpuLayers)
                    LlamaLanguageModelExitTests.startGeneration(session) {
                        LlamaLanguageModelExitTests.holdUntilShutdownStarts()
                    }
                    ExitTestTimer.stepStarted(name)
                    withExtendedLifetime(model) { exit(0) }
                }
                await ExitTestTimer.finish(name, watchdog, measuring: false)
            }

            @Test("A generation that runs past the deadline doesn't keep the process alive")
            func exitPastDeadline() async throws {
                let name = "exitPastDeadline"
                let watchdog = ExitTestTimer.watch(name)
                // The handler frees nothing, so ggml's assertion ends the process,
                // as it would without the handler.
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("exitPastDeadline")
                    let (model, session) = try await LlamaLanguageModelExitTests.loadedModel(gpuLayers: -1)
                    LlamaLanguageModelExitTests.startGeneration(session) {
                        while true { sleep(60) }
                    }
                    ExitTestTimer.stepStarted("exitPastDeadline")
                    withExtendedLifetime(model) { exit(0) }
                }
                let elapsed = try #require(await ExitTestTimer.finish(name, watchdog))
                #expect(elapsed >= .seconds(LlamaLifecycle.exitDeadline))
            }

            @Test("An exit from inside a generation doesn't wait for that generation")
            func exitInsideOperation() async throws {
                let name = "exitInsideOperation"
                let watchdog = ExitTestTimer.watch(name)
                // The handler returns at once and frees nothing,
                // so ggml's assertion ends the process before the deadline.
                // The exit status can't show how long the handler took;
                // `shutDownInsideTaskOperation()` checks that it returns at once.
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("exitInsideOperation")
                    let (model, session) = try await LlamaLanguageModelExitTests.loadedModel(gpuLayers: -1)
                    LlamaLanguageModelExitTests.startGeneration(session) {
                        ExitTestTimer.stepStarted("exitInsideOperation")
                        exit(0)
                    }
                    // The generation's thread exits the process.
                    withExtendedLifetime(model) {
                        while true { sleep(60) }
                    }
                }
                await ExitTestTimer.finish(name, watchdog, measuring: false)
            }

            @Test("A log callback that exits while a model is freed doesn't hang the process")
            func exitFromLogDuringCleanup() async throws {
                // The handler sees that the cleanup's thread is exiting, and returns at once,
                // which `shutDownInsideThreadOperation()` checks without a model.
                // The model is only partly freed, so ggml's assertion ends the process,
                // as it would without the handler.
                let name = "exitFromLogDuringCleanup"
                let watchdog = ExitTestTimer.watch(name)
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("exitFromLogDuringCleanup")
                    do {
                        let (model, session) = try await LlamaLanguageModelExitTests.loadedModel(gpuLayers: -1)
                        let once = Locked(false)
                        LlamaTestHooks.onLog.withLock {
                            $0 = {
                                let isFirst = once.withLock { done in
                                    defer { done = true }
                                    return !done
                                }
                                if isFirst { exit(0) }
                            }
                        }
                        ExitTestTimer.stepStarted("exitFromLogDuringCleanup")
                        // Releasing the last references frees the model,
                        // and llama.cpp logs while it frees.
                        _ = (model, session)
                    }
                    // A distinct status shows that llama.cpp logged nothing.
                    exit(4)
                }
                await ExitTestTimer.finish(name, watchdog, measuring: false)
            }

            @Test("A cleanup that blocks at exit aborts the process instead of hanging")
            func exitWithBlockedCleanup() async throws {
                // A log callback blocks while the exit handler frees the model,
                // so the cleanup watchdog aborts the process.
                let name = "exitWithBlockedCleanup"
                let watchdog = ExitTestTimer.watch(name)
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("exitWithBlockedCleanup")
                    let (model, session) = try await LlamaLanguageModelExitTests.loadedModel(gpuLayers: -1)
                    LlamaTestHooks.onLog.withLock {
                        $0 = {
                            if LlamaLifecycle.shared.isStopping {
                                while true { sleep(60) }
                            }
                        }
                    }
                    ExitTestTimer.stepStarted("exitWithBlockedCleanup")
                    withExtendedLifetime((model, session)) { exit(0) }
                }
                let elapsed = try #require(await ExitTestTimer.finish(name, watchdog))
                #expect(elapsed >= .seconds(LlamaLifecycle.cleanupDeadline))
            }
        #endif

        @Test("A process exits normally after the model is released", arguments: [Int32(-1), 0])
        func exitAfterRelease(gpuLayers: Int32) async {
            let name = "exitAfterRelease\(gpuLayers)"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) { [name = name as String, gpuLayers = gpuLayers as Int32] in
                ExitTestTimer.childStarted(name)
                weak var released: LlamaLanguageModel?
                do {
                    let (model, _) = try await LlamaLanguageModelExitTests.loadedModel(gpuLayers: gpuLayers)
                    released = model
                }
                // A distinct status shows that something still holds the model.
                guard released == nil else { exit(4) }
                exit(0)
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }

        @Test("After the exit handler runs, a request throws, and nothing is freed twice")
        func requestAfterShutdown() async {
            let name = "requestAfterShutdown"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) {
                ExitTestTimer.childStarted("requestAfterShutdown")
                weak var released: LlamaLanguageModel?
                do {
                    let (model, session) = try await LlamaLanguageModelExitTests.loadedModel(gpuLayers: -1)
                    released = model
                    guard LlamaLifecycle.shared.shutDown() == .freed else { exit(5) }
                    // The cached context is freed already, so this frees nothing.
                    model.clearCachedContext()
                    do {
                        _ = try await session.respond(to: "Say hello")
                        exit(6)
                    } catch LanguageModelSession.GenerationError.assetsUnavailable {}
                    do {
                        for try await _ in session.streamResponse(to: "Say hello") {}
                        exit(7)
                    } catch LanguageModelSession.GenerationError.assetsUnavailable {}
                }
                // The model's deinit ran after the handler freed the model,
                // and freed nothing again.
                guard released == nil else { exit(4) }
                exit(0)
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }

        @Test("After the exit handler runs, a model that isn't loaded fails to load")
        func loadAfterShutdown() async {
            let name = "loadAfterShutdown"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) {
                ExitTestTimer.childStarted("loadAfterShutdown")
                _ = try await LlamaLanguageModelExitTests.loadedModel(gpuLayers: -1)
                LlamaLifecycle.shared.shutDown()
                let other = LlamaLanguageModel(modelPath: LlamaLanguageModelExitTests.modelPath)
                do {
                    _ = try await LanguageModelSession(model: other).respond(to: "Say hello")
                    exit(5)
                } catch LanguageModelSession.GenerationError.assetsUnavailable {}
                // A missing file throws the exit error, not an invalid path.
                let missing = LlamaLanguageModel(modelPath: "/nonexistent/model.gguf")
                do {
                    _ = try await LanguageModelSession(model: missing).respond(to: "Say hello")
                    exit(6)
                } catch LanguageModelSession.GenerationError.assetsUnavailable {}
                exit(0)
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }
    }

    /// Lets the parent of an exit test kill a child that hangs,
    /// and measure how long the child takes to end after the step under test.
    ///
    /// Swift Testing's time limit doesn't end a child process,
    /// so a child that hangs would keep the test run waiting.
    /// The child writes its process ID when it starts,
    /// and the monotonic time when the step under test starts,
    /// to files that both processes name from the parent's process ID.
    /// The measurement starts after the child's setup,
    /// so a slow load doesn't change it.
    ///
    /// The measurement ends when the parent sees that the child has ended,
    /// which a busy machine can delay by seconds.
    /// So a test checks only that a time is at least a bound,
    /// which a delay can't break,
    /// and never that a time is less than a bound.
    enum ExitTestTimer {
        /// How long the parent waits for a child before it kills the child.
        static let limit: Duration = .seconds(120)

        /// The current time of a monotonic clock that every process on the system shares.
        static func now() -> UInt64 {
            clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        }

        private static func file(_ name: String, _ suffix: String, parent: pid_t) -> URL {
            URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("llama-exit-test-\(parent)-\(name).\(suffix)")
        }

        /// The exit status of a child that couldn't write a timer file.
        ///
        /// The child also writes the error to standard error.
        static let writeFailureStatus: CInt = 70

        /// In the child: writes `text` to a timer file,
        /// or ends the child with ``writeFailureStatus`` if it can't.
        private static func write(_ text: String, to url: URL) {
            do {
                try text.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                fputs("ExitTestTimer couldn't write \(url.path): \(error)\n", stderr)
                _exit(writeFailureStatus)
            }
        }

        /// In the child: records its process ID.
        static func childStarted(_ name: String) {
            write(String(getpid()), to: file(name, "pid", parent: getppid()))
        }

        /// In the child: records when the step under test starts.
        static func stepStarted(_ name: String) {
            write(String(now()), to: file(name, "start", parent: getppid()))
        }

        private static func read(_ name: String, _ suffix: String) -> String? {
            try? String(contentsOf: file(name, suffix, parent: getpid()), encoding: .utf8)
        }

        /// In the parent: starts a task that kills the child if it hasn't ended after ``limit``,
        /// and returns whether it did.
        static func watch(_ name: String) -> Task<Bool, Never> {
            for suffix in ["pid", "start"] {
                try? FileManager.default.removeItem(at: file(name, suffix, parent: getpid()))
            }
            return Task.detached {
                guard (try? await Task.sleep(for: limit)) != nil else { return false }
                guard let pid = read(name, "pid").flatMap({ pid_t($0) }) else { return false }
                kill(pid, SIGKILL)
                return true
            }
        }

        /// In the parent, after the child ends: stops the watchdog,
        /// records an issue if it killed the child,
        /// and, if `measuring`, returns the time from the step's start to now.
        ///
        /// If the start time can't be read,
        /// this records an issue with the error and returns `nil`.
        /// A child that couldn't write the file
        /// exits with ``writeFailureStatus`` instead.
        @discardableResult
        static func finish(
            _ name: String,
            _ watchdog: Task<Bool, Never>,
            measuring: Bool = true
        ) async -> Duration? {
            watchdog.cancel()
            let killed = await watchdog.value
            #expect(!killed, "The child process didn't end within \(limit).")
            guard measuring else { return nil }
            let url = file(name, "start", parent: getpid())
            let text: String
            do {
                text = try String(contentsOf: url, encoding: .utf8)
            } catch {
                Issue.record("Couldn't read the step's start time from \(url.path): \(error)")
                return nil
            }
            guard let start = UInt64(text) else {
                Issue.record("The step's start time in \(url.path) isn't a number: \(text.debugDescription)")
                return nil
            }
            return .nanoseconds(Int64(now() - start))
        }
    }

    /// Tests of the exit handler's bookkeeping, without a model.
    @Suite("LlamaLifecycle")
    struct LlamaLifecycleTests {
        @Test("Shutting down with no work frees, and a second shutdown does nothing")
        func shutDownIdle() throws {
            let lifecycle = LlamaLifecycle()
            #expect(lifecycle.shutDown(deadline: 0) == .freed)
            #expect(lifecycle.shutDown(deadline: 0) == .alreadyShutDown)
            #expect(lifecycle.isStopping)
            #expect(throws: LanguageModelSession.GenerationError.self) { try lifecycle.begin() }
            #expect(throws: LanguageModelSession.GenerationError.self) { try lifecycle.perform {} }
        }

        @Test("Shutting down waits for running work, and frees nothing at the deadline")
        func shutDownTimesOut() async throws {
            let lifecycle = LlamaLifecycle()
            nonisolated(unsafe) let operation = try lifecycle.begin()
            // Shut down from a thread that runs no task, so the operation isn't its own.
            let result = await withCheckedContinuation { continuation in
                Thread.detachNewThread {
                    let start = ContinuousClock.now
                    let result = lifecycle.shutDown(deadline: 0.2)
                    continuation.resume(returning: (result, ContinuousClock.now - start))
                }
            }
            #expect(result.0 == .timedOut)
            #expect(result.1 >= .milliseconds(200))
            operation.end()
            #expect(lifecycle.activeOperationCount == 0)
        }

        @Test("Shutting down returns when running work ends")
        func shutDownWaitsForWork() async throws {
            let lifecycle = LlamaLifecycle()
            nonisolated(unsafe) let operation = try lifecycle.begin()
            let result = await withCheckedContinuation { continuation in
                Thread.detachNewThread {
                    continuation.resume(returning: lifecycle.shutDown(deadline: 10))
                }
                usleep(100_000)
                operation.end()
            }
            #expect(result == .freed)
        }

        @Test("Shutting down inside an operation of the current task returns at once")
        func shutDownInsideTaskOperation() throws {
            let lifecycle = LlamaLifecycle()
            let operation = try lifecycle.begin()
            #expect(lifecycle.shutDown(deadline: 10) == .insideOperation)
            operation.end()
        }

        @Test("Shutting down inside synchronous work on the current thread returns at once")
        func shutDownInsideThreadOperation() throws {
            let lifecycle = LlamaLifecycle()
            let result = try lifecycle.perform { lifecycle.shutDown(deadline: 10) }
            #expect(result == .insideOperation)
        }

        @Test("A suspended operation doesn't hold up a shutdown, and can't resume after it")
        func suspendedOperation() throws {
            let lifecycle = LlamaLifecycle()
            let operation = try lifecycle.begin()
            operation.suspend()
            #expect(lifecycle.activeOperationCount == 0)
            #expect(lifecycle.shutDown(deadline: 0) == .freed)
            #expect(throws: LanguageModelSession.GenerationError.self) { try operation.resume() }
            operation.end()
            #expect(lifecycle.activeOperationCount == 0)
        }

        @Test("The cleanup deadline does nothing after the cleanup ends")
        func cleanupDeadlineAfterCleanup() async throws {
            let name = "cleanupDeadlineAfterCleanup"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) {
                ExitTestTimer.childStarted("cleanupDeadlineAfterCleanup")
                ExitTestTimer.stepStarted("cleanupDeadlineAfterCleanup")
                let value = LlamaLifecycle.withAbortDeadline(0.1) { 42 }
                // Outlive the deadline; the watchdog must not abort.
                usleep(500_000)
                exit(value == 42 ? 0 : 5)
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }

        @Test("A cleanup that runs past its deadline aborts the process")
        func cleanupDeadlinePassed() async throws {
            let name = "cleanupDeadlinePassed"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .signal(SIGABRT)) {
                ExitTestTimer.childStarted("cleanupDeadlinePassed")
                ExitTestTimer.stepStarted("cleanupDeadlinePassed")
                LlamaLifecycle.withAbortDeadline(0.2) {
                    while true { sleep(60) }
                }
            }
            let elapsed = try #require(await ExitTestTimer.finish(name, watchdog))
            #expect(elapsed >= .milliseconds(200))
        }

        @Test("A cleanup that runs past its deadline aborts even if standard error is a full pipe")
        func cleanupDeadlineWithFullStandardError() async throws {
            let name = "cleanupDeadlineWithFullStandardError"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .signal(SIGABRT)) {
                ExitTestTimer.childStarted("cleanupDeadlineWithFullStandardError")
                // Make standard error a pipe that is full and that nobody reads.
                // The read end stays open, so a write blocks instead of failing.
                var fds: [CInt] = [0, 0]
                guard pipe(&fds) == 0 else { _exit(3) }
                let flags = fcntl(fds[1], F_GETFL)
                _ = fcntl(fds[1], F_SETFL, flags | O_NONBLOCK)
                let chunk = [UInt8](repeating: 0x2E, count: 4096)
                while chunk.withUnsafeBytes({ Darwin.write(fds[1], $0.baseAddress, $0.count) }) > 0 {}
                guard errno == EAGAIN else { _exit(4) }
                _ = fcntl(fds[1], F_SETFL, flags)
                guard dup2(fds[1], STDERR_FILENO) == STDERR_FILENO else { _exit(6) }
                ExitTestTimer.stepStarted("cleanupDeadlineWithFullStandardError")
                LlamaLifecycle.withAbortDeadline(0.2) {
                    while true { sleep(60) }
                }
            }
            let elapsed = try #require(await ExitTestTimer.finish(name, watchdog))
            #expect(elapsed >= .milliseconds(200))
        }

        #if DEBUG
            @Test("Once the cleanup completes, a watchdog that timed out doesn't abort")
            func cleanupCompletionWinsRace() async throws {
                let name = "cleanupCompletionWinsRace"
                let watchdog = ExitTestTimer.watch(name)
                await #expect(processExitsWith: .success) {
                    ExitTestTimer.childStarted("cleanupCompletionWinsRace")
                    let entered = DispatchSemaphore(value: 0)
                    let proceed = DispatchSemaphore(value: 0)
                    let decided = DispatchSemaphore(value: 0)
                    let aborts = Locked<Bool?>(nil)
                    LlamaTestHooks.cleanupWatchdog.withLock {
                        $0 = { event in
                            switch event {
                            case .beforeDecision:
                                // Hold the watchdog after its timeout
                                // until the cleanup has completed.
                                entered.signal()
                                proceed.wait()
                            case .decided(let value):
                                aborts.withLock { $0 = value }
                                decided.signal()
                            }
                        }
                    }
                    ExitTestTimer.stepStarted("cleanupCompletionWinsRace")
                    // The cleanup returns only after the watchdog's deadline has passed.
                    let value = LlamaLifecycle.withAbortDeadline(0.05) {
                        entered.wait()
                        return 42
                    }
                    proceed.signal()
                    decided.wait()
                    exit(value == 42 && aborts.withLock({ $0 }) == false ? 0 : 5)
                }
                await ExitTestTimer.finish(name, watchdog, measuring: false)
            }

            @Test("Once a watchdog decides to abort, the cleanup doesn't return")
            func cleanupTimeoutWinsRace() async throws {
                let name = "cleanupTimeoutWinsRace"
                let watchdog = ExitTestTimer.watch(name)
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("cleanupTimeoutWinsRace")
                    let decided = DispatchSemaphore(value: 0)
                    LlamaTestHooks.cleanupWatchdog.withLock {
                        $0 = { event in
                            guard event == .decided(aborts: true) else { return }
                            decided.signal()
                            // Keep the watchdog from aborting,
                            // so that only the cleanup's own check can abort.
                            while true { sleep(60) }
                        }
                    }
                    ExitTestTimer.stepStarted("cleanupTimeoutWinsRace")
                    _ = LlamaLifecycle.withAbortDeadline(0.05) {
                        decided.wait()
                        return 42
                    }
                    // A distinct status shows that the cleanup returned.
                    exit(5)
                }
                await ExitTestTimer.finish(name, watchdog, measuring: false)
            }
        #endif

        @Test("A failed registration of the exit handler makes the handler free nothing")
        func registrationFailure() {
            let lifecycle = LlamaLifecycle()
            lifecycle.exitHandlerRegistrationFailed()
            #expect(lifecycle.shutDown(deadline: 0) == .registrationFailed)
        }
    }
#endif  // Llama
