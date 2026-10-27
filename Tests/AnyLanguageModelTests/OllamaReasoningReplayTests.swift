import Foundation
import HTTPTypes
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

/// A reasoning model's reasoning goes back to it with each tool round, as Ollama takes it, so it
/// goes on from what it reasoned rather than starting again.
@Suite("Ollama reasoning replay")
struct OllamaReasoningReplayTests {
    @Test(
        "a tool round's thinking goes back with its tool calls, and is kept in the transcript",
        arguments: [false, true],
        [false, true]
    )
    func thinkingGoesBackWithToolCalls(streaming: Bool, profile: Bool) async throws {
        let call =
            #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"","thinking":"I should check the weather.","tool_calls":[{"id":"call_reasoning","type":"function","function":{"name":"getWeather","arguments":{"city":"Munich"}}}]},"done":true}"#
        let answer =
            #"{"model":"m","created_at":"2026-10-16T10:00:01.000Z","message":{"role":"assistant","content":"It's sunny."},"done":true}"#
        var rounds =
            streaming
            ? [
                [
                    #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"","thinking":"I should check the weather."},"done":false}"#,
                    #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_reasoning","type":"function","function":{"name":"getWeather","arguments":{"city":"Munich"}}}]},"done":false}"#,
                    #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":""},"done":true}"#,
                ],
                [answer],
            ] : [[call], [answer]]
        rounds.append([answer])
        let transport = ollama(rounds)
        let model = OllamaLanguageModel(
            baseURL: URL(string: "http://localhost:11434")!,
            model: "m",
            transport: transport
        )
        let completed = Locked(false)
        let callbacks = Locked<[Bool]>([])
        func makeSession(history: Transcript = .init()) -> LanguageModelSession {
            profile
                ? LanguageModelSession(
                    profile: OllamaReasoningProfile(model: model, completed: completed, callbacks: callbacks),
                    history: history
                ) : LanguageModelSession(model: model, tools: [WeatherTool()], transcript: history)
        }
        let session = makeSession()
        let response: LanguageModelSession.Response<String>
        if streaming {
            response = try await session.streamResponse(to: "Weather in Munich?").collect()
        } else {
            response = try await session.respond(to: "Weather in Munich?")
        }
        #expect(response.content == "It's sunny.")

        try #require(transport.bodies.count == 2)
        let messages = try #require(
            (try JSONSerialization.jsonObject(with: transport.bodies[1]) as? [String: Any])?["messages"]
                as? [[String: Any]]
        )
        let asked = try #require(messages.first { $0["tool_calls"] != nil })
        #expect(asked["thinking"] as? String == "I should check the weather.")
        let reasoning = session.transcript.compactMap { entry -> Transcript.Reasoning? in
            if case .reasoning(let reasoning) = entry { return reasoning }
            return nil
        }
        #expect(reasoning.map(\.origin) == [.ollama])
        if profile { #expect(callbacks.withLock { $0 } == [true]) }

        let restored = try JSONDecoder().decode(Transcript.self, from: JSONEncoder().encode(session.transcript))
        _ = try await makeSession(history: restored).respond(to: "And tomorrow?")
        try #require(transport.bodies.count == 3)
        let replay = try #require(
            (try JSONSerialization.jsonObject(with: transport.bodies[2]) as? [String: Any])?["messages"]
                as? [[String: Any]]
        )
        let replayed = try #require(replay.first { $0["tool_calls"] != nil })
        #expect(NSDictionary(dictionary: asked).isEqual(to: replayed))
        if profile { #expect(callbacks.withLock { $0 } == [true]) }
    }

    @Test(
        "the round past the tool-round limit leaves its thinking out of the rounds that ran",
        arguments: [false, true],
        [false, true]
    )
    func toolRoundLimitOmitsTheThinkingOfTheRoundPastIt(streaming: Bool, profile: Bool) async throws {
        let round =
            streaming
            ? [
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"","thinking":"I should check the weather."},"done":false}"#,
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_reasoning","type":"function","function":{"name":"getWeather","arguments":{"city":"Munich"}}}]},"done":false}"#,
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":""},"done":true}"#,
            ]
            : [
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"","thinking":"I should check the weather.","tool_calls":[{"id":"call_reasoning","type":"function","function":{"name":"getWeather","arguments":{"city":"Munich"}}}]},"done":true}"#
            ]
        let model = OllamaLanguageModel(
            baseURL: URL(string: "http://localhost:11434")!,
            model: "m",
            transport: ollama([round, round])
        )
        var options = GenerationOptions()
        options.maximumToolRounds = 1
        let session =
            profile
            ? LanguageModelSession(profile: LanguageModelSession.Profile { WeatherTool() }.model(model))
            : LanguageModelSession(model: model, tools: [WeatherTool()])
        do {
            if streaming {
                _ = try await session.streamResponse(to: "Weather in Munich?", options: options).collect()
            } else {
                _ = try await session.respond(to: "Weather in Munich?", options: options)
            }
            Issue.record("Expected the tool-round limit")
        } catch let error as LanguageModelSession.ToolRoundLimitExceeded {
            let kinds = error.transcriptEntries.map { entry in
                switch entry {
                case .reasoning: "reasoning"
                case .toolCalls: "toolCalls"
                case .toolOutput: "toolOutput"
                default: "other"
                }
            }
            #expect(kinds == ["reasoning", "toolCalls", "toolOutput"])
        }
    }

    @Test("a streamed answer's own reasoning is in its last snapshot")
    func finalRoundReasoningIsInTheLastSnapshot() async throws {
        let transport = ollama([
            [
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"","thinking":"Just greet."},"done":false}"#,
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"Hi!"},"done":false}"#,
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":""},"done":true}"#,
            ]
        ])
        let model = OllamaLanguageModel(
            baseURL: URL(string: "http://localhost:11434")!,
            model: "m",
            transport: transport
        )
        var last: LanguageModelSession.ResponseStream<String>.Snapshot?
        for try await snapshot in LanguageModelSession(model: model).streamResponse(to: "Hi") { last = snapshot }
        let entries = try #require(last).transcriptEntries
        #expect(
            entries.contains { if case .reasoning(let reasoning) = $0 { reasoning.origin == .ollama } else { false } }
        )
    }

    @Test("reasoning in the transcript goes back with the assistant message it came before")
    func transcriptReasoningGoesBack() async throws {
        let transport = ollama([
            [
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"Fine."},"done":true}"#
            ]
        ])
        let model = OllamaLanguageModel(
            baseURL: URL(string: "http://localhost:11434")!,
            model: "m",
            transport: transport
        )
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: "Hi"))])),
            .reasoning(.init(origin: .ollama, segments: [.text(.init(content: "Greet back."))])),
            .response(.init(assetIDs: [], segments: [.text(.init(content: "Hello."))])),
        ])
        let session = LanguageModelSession(model: model, transcript: transcript)
        _ = try await session.respond(to: "How are you?")
        let body = try #require(transport.bodies.first)
        let messages = try #require(
            (try JSONSerialization.jsonObject(with: body) as? [String: Any])?["messages"] as? [[String: Any]]
        )
        let answered = try #require(messages.first { $0["role"] as? String == "assistant" })
        #expect(answered["thinking"] as? String == "Greet back.")
        #expect(!messages.contains { $0["role"] as? String == "user" && $0["thinking"] != nil })
    }

    /// Reasoning that no assistant message followed, as a turn that failed or was cancelled leaves
    /// it with `.preserveTranscript`, isn't sent with the next turn's answer: a prompt or
    /// instructions start anew.
    @Test(
        "reasoning a failed turn left goes back with nothing",
        arguments: [
            Transcript.Entry.prompt(.init(segments: [.text(.init(content: "Again"))])),
            .instructions(.init(segments: [.text(.init(content: "Be brief."))], toolDefinitions: [])),
        ]
    )
    func reasoningAFailedTurnLeftIsntSent(startingAnew: Transcript.Entry) async throws {
        let transport = ollama([
            [
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"Fine."},"done":true}"#
            ]
        ])
        let model = OllamaLanguageModel(
            baseURL: URL(string: "http://localhost:11434")!,
            model: "m",
            transport: transport
        )
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: "Hi"))])),
            .reasoning(.init(origin: .ollama, segments: [.text(.init(content: "Left over."))])),
            startingAnew,
            .response(.init(assetIDs: [], segments: [.text(.init(content: "Hello."))])),
        ])
        let session = LanguageModelSession(model: model, transcript: transcript)
        _ = try await session.respond(to: "How are you?")
        let body = try #require(transport.bodies.first)
        let messages = try #require(
            (try JSONSerialization.jsonObject(with: body) as? [String: Any])?["messages"] as? [[String: Any]]
        )
        #expect(!messages.contains { $0["thinking"] != nil }, "\(messages)")
    }
}

private struct OllamaReasoningProfile: LanguageModelSession.DynamicProfile {
    let model: OllamaLanguageModel
    let completed: Locked<Bool>
    let callbacks: Locked<[Bool]>

    var body: some LanguageModelSession.DynamicProfile {
        let beforeTool = completed.withLock { !$0 }
        return LanguageModelSession.Profile { WeatherTool() }
            .model(model)
            .onReasoning { _ in callbacks.withLock { $0.append(beforeTool) } }
            .onToolOutput { _, _ in completed.withLock { $0 = true } }
    }
}

/// A transport whose replies are rounds of Ollama's JSON lines.
private func ollama(_ rounds: [[String]]) -> FakeTransport {
    FakeTransport(replies: rounds.map { .init(chunks: [$0.joined(separator: "\n") + "\n"]) })
}
