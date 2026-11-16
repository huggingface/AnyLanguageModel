import Foundation
import Testing

@testable import AnyLanguageModel

#if Llama
    import LlamaSwift

    /// Tests of which llama.cpp log messages the log callback writes.
    ///
    /// Each test starts with a message that isn't a continuation,
    /// which sets the continuation state of its thread and logger.
    @Suite("Llama log filter")
    struct LlamaLogFilterTests {
        typealias Runtime = LlamaRuntime

        @Test("A message is written at or above the minimum level")
        func threshold() {
            #expect(Runtime.shouldWrite(level: GGML_LOG_LEVEL_WARN, logger: .llama, minimum: .warn))
            #expect(Runtime.shouldWrite(level: GGML_LOG_LEVEL_ERROR, logger: .llama, minimum: .warn))
            #expect(!Runtime.shouldWrite(level: GGML_LOG_LEVEL_INFO, logger: .llama, minimum: .warn))
            #expect(Runtime.shouldWrite(level: GGML_LOG_LEVEL_DEBUG, logger: .llama, minimum: .debug))
        }

        @Test("A continuation follows the message that it continues")
        func continuationFollowsMessage() {
            #expect(Runtime.shouldWrite(level: GGML_LOG_LEVEL_WARN, logger: .llama, minimum: .warn))
            #expect(Runtime.shouldWrite(level: GGML_LOG_LEVEL_CONT, logger: .llama, minimum: .warn))
            #expect(!Runtime.shouldWrite(level: GGML_LOG_LEVEL_INFO, logger: .llama, minimum: .warn))
            #expect(!Runtime.shouldWrite(level: GGML_LOG_LEVEL_CONT, logger: .llama, minimum: .warn))
        }

        @Test("A continuation keeps its message's decision when the level changes in between")
        func thresholdChangeBeforeContinuation() {
            #expect(Runtime.shouldWrite(level: GGML_LOG_LEVEL_WARN, logger: .llama, minimum: .warn))
            #expect(Runtime.shouldWrite(level: GGML_LOG_LEVEL_CONT, logger: .llama, minimum: .error))
            #expect(!Runtime.shouldWrite(level: GGML_LOG_LEVEL_INFO, logger: .llama, minimum: .warn))
            #expect(!Runtime.shouldWrite(level: GGML_LOG_LEVEL_CONT, logger: .llama, minimum: .debug))
        }

        @Test("Nothing is written at the none level, including continuations")
        func none() {
            for level in [GGML_LOG_LEVEL_DEBUG, GGML_LOG_LEVEL_INFO, GGML_LOG_LEVEL_WARN, GGML_LOG_LEVEL_ERROR] {
                #expect(!Runtime.shouldWrite(level: level, logger: .llama, minimum: .none))
            }
            // A warning that was written, then a change to `.none`, then its continuation.
            #expect(Runtime.shouldWrite(level: GGML_LOG_LEVEL_WARN, logger: .llama, minimum: .warn))
            #expect(!Runtime.shouldWrite(level: GGML_LOG_LEVEL_CONT, logger: .llama, minimum: .none))
            // A message filtered at `.none`, then a change back, then its continuation.
            #expect(!Runtime.shouldWrite(level: GGML_LOG_LEVEL_DEBUG, logger: .llama, minimum: .none))
            #expect(!Runtime.shouldWrite(level: GGML_LOG_LEVEL_CONT, logger: .llama, minimum: .warn))
        }

        @Test("Continuations from the two loggers on one thread follow their own messages")
        func interleavedLoggers() {
            #expect(Runtime.shouldWrite(level: GGML_LOG_LEVEL_WARN, logger: .llama, minimum: .warn))
            #expect(!Runtime.shouldWrite(level: GGML_LOG_LEVEL_INFO, logger: .projector, minimum: .warn))
            #expect(Runtime.shouldWrite(level: GGML_LOG_LEVEL_CONT, logger: .llama, minimum: .warn))
            #expect(!Runtime.shouldWrite(level: GGML_LOG_LEVEL_CONT, logger: .projector, minimum: .warn))
        }

        @Test("Continuations from two threads follow the messages of their own thread")
        func interleavedThreads() {
            let first = DispatchSemaphore(value: 0)
            let second = DispatchSemaphore(value: 0)
            let done = DispatchSemaphore(value: 0)
            let results = Locked<[String: Bool]>([:])
            // Thread A writes a warning, then thread B filters an info message,
            // then each continues its own message.
            Thread.detachNewThread {
                let message = Runtime.shouldWrite(level: GGML_LOG_LEVEL_WARN, logger: .llama, minimum: .warn)
                first.signal()
                second.wait()
                let continuation = Runtime.shouldWrite(level: GGML_LOG_LEVEL_CONT, logger: .llama, minimum: .warn)
                results.withLock { $0["A"] = message && continuation }
                done.signal()
            }
            Thread.detachNewThread {
                first.wait()
                let message = Runtime.shouldWrite(level: GGML_LOG_LEVEL_INFO, logger: .llama, minimum: .warn)
                second.signal()
                let continuation = Runtime.shouldWrite(level: GGML_LOG_LEVEL_CONT, logger: .llama, minimum: .warn)
                results.withLock { $0["B"] = message || continuation }
                done.signal()
            }
            done.wait()
            done.wait()
            #expect(results.withLock { $0 } == ["A": true, "B": false])
        }
    }
#endif  // Llama
