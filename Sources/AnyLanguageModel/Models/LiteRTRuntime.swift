import Foundation

#if LiteRT && (os(iOS) || os(macOS)) && !targetEnvironment(macCatalyst)
    @preconcurrency import LiteRTLM

    protocol LiteRTRuntime: Sendable {
        func makeConversation(config: ConversationConfig) async throws -> any LiteRTConversation
    }

    protocol LiteRTConversation: Sendable {
        func sendMessageStream(_ message: Message, maxOutputTokens: Int?) -> AsyncThrowingStream<Message, Error>
        func cancel() throws
    }

    extension Engine: LiteRTRuntime {
        func makeConversation(config: ConversationConfig) throws -> any LiteRTConversation {
            NativeLiteRTConversation(conversation: try createConversation(with: config))
        }
    }

    private struct NativeLiteRTConversation: LiteRTConversation {
        let conversation: Conversation

        func sendMessageStream(_ message: Message, maxOutputTokens: Int?) -> AsyncThrowingStream<Message, Error> {
            conversation.sendMessageStream(message, maxOutputTokens: maxOutputTokens)
        }

        func cancel() throws {
            try conversation.cancel()
        }
    }

    /// Shares successful loading and in-flight work across requests.
    /// A failed attempt is cleared only by its own waiters,
    /// so an older failure cannot discard a newer retry.
    actor LiteRTModelLoader {
        private var attempt: (id: UUID, task: Task<any LiteRTRuntime, Error>)?
        private let load: @Sendable () async throws -> any LiteRTRuntime

        init(_ load: @escaping @Sendable () async throws -> any LiteRTRuntime) {
            self.load = load
        }

        func ready() async throws -> any LiteRTRuntime {
            try Task.checkCancellation()
            if attempt == nil {
                let load = self.load
                attempt = (UUID(), Task { try await load() })
            }
            let current = attempt!
            let runtime: any LiteRTRuntime
            do {
                runtime = try await current.task.value
            } catch {
                if attempt?.id == current.id {
                    attempt = nil
                }
                throw error
            }
            try Task.checkCancellation()
            return runtime
        }
    }

    /// Serializes native inference startup with cancellation.
    /// Cancelling before startup prevents inference from starting;
    /// cancelling after startup stops the active conversation.
    private final class LiteRTGeneration: Sendable {
        private struct State: Sendable {
            var isCancelled = false
            var conversation: (any LiteRTConversation)?
        }

        private let state = Locked(State())

        func start(
            conversation: any LiteRTConversation,
            prompt: Message,
            maximumResponseTokens: Int?
        ) throws -> AsyncThrowingStream<Message, Error> {
            try state.withLock { state in
                guard !state.isCancelled else { throw CancellationError() }
                let stream = conversation.sendMessageStream(prompt, maxOutputTokens: maximumResponseTokens)
                state.conversation = conversation
                return stream
            }
        }

        func cancel() {
            let conversation = state.withLock { state in
                state.isCancelled = true
                let conversation = state.conversation
                state.conversation = nil
                return conversation
            }
            try? conversation?.cancel()
        }

        func finish() {
            state.withLock { $0.conversation = nil }
        }
    }

    func generateLiteRTResponse(
        conversation: any LiteRTConversation,
        prompt: Message,
        maximumResponseTokens: Int?,
        onChunk: (String) throws -> Void
    ) async throws {
        let generation = LiteRTGeneration()
        try await withTaskCancellationHandler {
            defer {
                if Task.isCancelled { generation.cancel() }
                generation.finish()
            }
            try Task.checkCancellation()
            let stream = try generation.start(
                conversation: conversation,
                prompt: prompt,
                maximumResponseTokens: maximumResponseTokens
            )
            for try await chunk in stream {
                try Task.checkCancellation()
                try onChunk(chunk.toString)
            }
            try Task.checkCancellation()
        } onCancel: {
            generation.cancel()
        }
    }
#endif
