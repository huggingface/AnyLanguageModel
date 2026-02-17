import Testing

@testable import AnyLanguageModel

private actor ToolExecutionDelegateSpy: ToolExecutionDelegate {
    private(set) var generatedToolCalls: [Transcript.ToolCall] = []
    private(set) var executedToolCalls: [Transcript.ToolCall] = []
    private(set) var executedOutputs: [Transcript.ToolOutput] = []
    private(set) var failures: [any Error] = []

    private let decisionProvider: @Sendable (Transcript.ToolCall) async -> ToolExecutionDecision

    init(decisionProvider: @escaping @Sendable (Transcript.ToolCall) async -> ToolExecutionDecision) {
        self.decisionProvider = decisionProvider
    }

    func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: LanguageModelSession) async {
        generatedToolCalls.append(contentsOf: toolCalls)
    }

    func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: LanguageModelSession
    ) async -> ToolExecutionDecision {
        await decisionProvider(toolCall)
    }

    func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: LanguageModelSession
    ) async {
        executedToolCalls.append(toolCall)
        executedOutputs.append(output)
    }

    func didFailToolCall(
        _ toolCall: Transcript.ToolCall,
        error: any Error,
        in session: LanguageModelSession
    ) async {
        failures.append(error)
    }
}

private struct ThrowingTool: Tool {
    let name = "throwingTool"
    let description = "A tool that throws"
    @Generable
    struct Arguments {
        @Guide(description: "Ignored")
        var message: String
    }
    func call(arguments: Arguments) async throws -> String {
        throw ThrowingToolError.testError
    }
}

private enum ThrowingToolError: Error, Equatable { case testError }

private struct DefaultToolExecutionDelegate: ToolExecutionDelegate {}

private struct ToolCallingTestModel: LanguageModel {
    typealias UnavailableReason = Never

    let toolCalls: [Transcript.ToolCall]
    let responseText: String

    init(toolCalls: [Transcript.ToolCall], responseText: String = "done") {
        self.toolCalls = toolCalls
        self.responseText = responseText
    }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        guard type == String.self else {
            fatalError("ToolCallingTestModel only supports generating String content")
        }

        var entries: [Transcript.Entry] = []

        if !toolCalls.isEmpty {
            if let delegate = session.toolExecutionDelegate {
                await delegate.didGenerateToolCalls(toolCalls, in: session)
            }

            var decisions: [ToolExecutionDecision] = []
            decisions.reserveCapacity(toolCalls.count)

            if let delegate = session.toolExecutionDelegate {
                for call in toolCalls {
                    let decision = await delegate.toolCallDecision(for: call, in: session)
                    if case .stop = decision {
                        entries.append(.toolCalls(Transcript.ToolCalls(toolCalls)))
                        return LanguageModelSession.Response(
                            content: "" as! Content,
                            rawContent: GeneratedContent(""),
                            transcriptEntries: ArraySlice(entries)
                        )
                    }
                    decisions.append(decision)
                }
            } else {
                decisions = Array(repeating: .execute, count: toolCalls.count)
            }

            entries.append(.toolCalls(Transcript.ToolCalls(toolCalls)))

            var toolsByName: [String: any Tool] = [:]
            for tool in session.tools {
                if toolsByName[tool.name] == nil {
                    toolsByName[tool.name] = tool
                }
            }

            for (index, call) in toolCalls.enumerated() {
                switch decisions[index] {
                case .stop:
                    // This branch should be unreachable because `.stop` returns during decision collection.
                    // Keep it as a defensive guard in case that logic changes.
                    entries = [.toolCalls(Transcript.ToolCalls(toolCalls))]
                    return LanguageModelSession.Response(
                        content: "" as! Content,
                        rawContent: GeneratedContent(""),
                        transcriptEntries: ArraySlice(entries)
                    )
                case .provideOutput(let segments):
                    let output = Transcript.ToolOutput(
                        id: call.id,
                        toolName: call.toolName,
                        segments: segments
                    )
                    if let delegate = session.toolExecutionDelegate {
                        await delegate.didExecuteToolCall(call, output: output, in: session)
                    }
                    entries.append(.toolOutput(output))
                case .execute:
                    guard let tool = toolsByName[call.toolName] else {
                        let message = Transcript.Segment.text(.init(content: "Tool not found: \(call.toolName)"))
                        let output = Transcript.ToolOutput(
                            id: call.id,
                            toolName: call.toolName,
                            segments: [message]
                        )
                        if let delegate = session.toolExecutionDelegate {
                            await delegate.didExecuteToolCall(call, output: output, in: session)
                        }
                        entries.append(.toolOutput(output))
                        continue
                    }

                    do {
                        let segments = try await tool.makeOutputSegments(from: call.arguments)
                        let output = Transcript.ToolOutput(
                            id: call.id,
                            toolName: tool.name,
                            segments: segments
                        )
                        if let delegate = session.toolExecutionDelegate {
                            await delegate.didExecuteToolCall(call, output: output, in: session)
                        }
                        entries.append(.toolOutput(output))
                    } catch {
                        if let delegate = session.toolExecutionDelegate {
                            await delegate.didFailToolCall(call, error: error, in: session)
                        }
                        throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
                    }
                }
            }
        }

        return LanguageModelSession.Response(
            content: responseText as! Content,
            rawContent: GeneratedContent(responseText),
            transcriptEntries: ArraySlice(entries)
        )
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        let rawContent = GeneratedContent(responseText)
        return LanguageModelSession.ResponseStream(content: responseText as! Content, rawContent: rawContent)
    }
}

@Suite("ToolExecutionDelegate")
struct ToolExecutionDelegateTests {
    @Test func defaultDelegateUsesExecuteDecisionAndNoOpCallbacks() async throws {
        let arguments = try GeneratedContent(json: #"{"city":"Cupertino"}"#)
        let toolCall = Transcript.ToolCall(id: "call-default", toolName: WeatherTool().name, arguments: arguments)
        let toolSpy = spy(on: WeatherTool())
        let session = LanguageModelSession(
            model: ToolCallingTestModel(toolCalls: [toolCall]),
            tools: [toolSpy]
        )
        session.toolExecutionDelegate = DefaultToolExecutionDelegate()

        let response = try await session.respond(to: "Hi")
        let calls = await toolSpy.calls

        #expect(calls.count == 1)
        #expect(
            response.transcriptEntries.contains { entry in
                if case .toolOutput = entry { return true }
                return false
            }
        )
    }

    @Test func defaultDelegateNoOpFailureCallbackDoesNotInterfereWithErrorPropagation() async throws {
        let arguments = try GeneratedContent(json: #"{"message":"fail"}"#)
        let toolCall = Transcript.ToolCall(id: "call-default-fail", toolName: ThrowingTool().name, arguments: arguments)
        let session = LanguageModelSession(
            model: ToolCallingTestModel(toolCalls: [toolCall]),
            tools: [ThrowingTool()]
        )
        session.toolExecutionDelegate = DefaultToolExecutionDelegate()

        await #expect(throws: LanguageModelSession.ToolCallError.self) {
            _ = try await session.respond(to: "Hi")
        }
    }

    @Test func toolExecutionDecisionCasesCanBeConstructed() {
        let execute = ToolExecutionDecision.execute
        let stop = ToolExecutionDecision.stop
        let provide = ToolExecutionDecision.provideOutput([.text(.init(content: "stub"))])

        if case .execute = execute {
            #expect(Bool(true))
        } else {
            Issue.record("Expected .execute")
        }

        if case .stop = stop {
            #expect(Bool(true))
        } else {
            Issue.record("Expected .stop")
        }

        if case .provideOutput(let segments) = provide {
            #expect(segments.count == 1)
        } else {
            Issue.record("Expected .provideOutput")
        }
    }

    @Test func stopAfterToolCalls() async throws {
        let arguments = try GeneratedContent(json: #"{"city":"Cupertino"}"#)
        let toolCall = Transcript.ToolCall(id: "call-1", toolName: WeatherTool().name, arguments: arguments)
        let delegate = ToolExecutionDelegateSpy { _ in .stop }
        let toolSpy = spy(on: WeatherTool())
        let session = LanguageModelSession(
            model: ToolCallingTestModel(toolCalls: [toolCall]),
            tools: [toolSpy]
        )
        session.toolExecutionDelegate = delegate

        let response = try await session.respond(to: "Hi")

        #expect(response.content.isEmpty)
        #expect(
            response.transcriptEntries.contains { entry in
                if case .toolCalls = entry { return true }
                return false
            }
        )
        #expect(
            !response.transcriptEntries.contains { entry in
                if case .toolOutput = entry { return true }
                return false
            }
        )

        let calls = await toolSpy.calls
        #expect(calls.isEmpty)

        let generatedCalls = await delegate.generatedToolCalls
        #expect(generatedCalls == [toolCall])
    }

    @Test func provideOutputBypassesExecution() async throws {
        let arguments = try GeneratedContent(json: #"{"city":"Cupertino"}"#)
        let toolCall = Transcript.ToolCall(id: "call-2", toolName: WeatherTool().name, arguments: arguments)
        let delegate = ToolExecutionDelegateSpy { _ in
            .provideOutput([.text(.init(content: "Stubbed"))])
        }
        let toolSpy = spy(on: WeatherTool())
        let session = LanguageModelSession(
            model: ToolCallingTestModel(toolCalls: [toolCall]),
            tools: [toolSpy]
        )
        session.toolExecutionDelegate = delegate

        let response = try await session.respond(to: "Hi")

        #expect(!response.transcriptEntries.isEmpty)
        #expect(
            response.transcriptEntries.contains { entry in
                if case let .toolOutput(output) = entry {
                    return output.segments.contains { segment in
                        if case .text(let text) = segment { return text.content == "Stubbed" }
                        return false
                    }
                }
                return false
            }
        )

        let calls = await toolSpy.calls
        #expect(calls.isEmpty)

        let executedOutputs = await delegate.executedOutputs
        #expect(executedOutputs.count == 1)
    }

    @Test func executeRunsToolAndNotifiesDelegate() async throws {
        let arguments = try GeneratedContent(json: #"{"city":"Cupertino"}"#)
        let toolCall = Transcript.ToolCall(id: "call-3", toolName: WeatherTool().name, arguments: arguments)
        let delegate = ToolExecutionDelegateSpy { _ in .execute }
        let toolSpy = spy(on: WeatherTool())
        let session = LanguageModelSession(
            model: ToolCallingTestModel(toolCalls: [toolCall]),
            tools: [toolSpy]
        )
        session.toolExecutionDelegate = delegate

        _ = try await session.respond(to: "Hi")

        let calls = await toolSpy.calls
        #expect(calls.count == 1)

        let executedCalls = await delegate.executedToolCalls
        #expect(executedCalls.count == 1)
    }

    @Test func didFailToolCallNotifiesDelegateWhenToolThrows() async throws {
        let arguments = try GeneratedContent(json: #"{"message":"fail"}"#)
        let toolCall = Transcript.ToolCall(id: "call-fail", toolName: ThrowingTool().name, arguments: arguments)
        let delegate = ToolExecutionDelegateSpy { _ in .execute }
        let session = LanguageModelSession(
            model: ToolCallingTestModel(toolCalls: [toolCall]),
            tools: [ThrowingTool()]
        )
        session.toolExecutionDelegate = delegate

        _ = try? await session.respond(to: "Hi")

        let failures = await delegate.failures
        #expect(failures.count == 1)
        #expect((failures.first as? ThrowingToolError) == .testError)
    }
}
