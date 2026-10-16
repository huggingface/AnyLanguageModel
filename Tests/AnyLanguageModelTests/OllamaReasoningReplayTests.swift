import Foundation
import HTTPTypes
import Testing

@testable import AnyLanguageModel

/// A reasoning model's reasoning goes back to it with each tool round, as Ollama takes it, so it
/// goes on from what it reasoned rather than starting again.
@Suite("Ollama reasoning replay")
struct OllamaReasoningReplayTests {
    @Test("a tool round's thinking goes back with its tool calls, and is kept in the transcript")
    func thinkingGoesBackWithToolCalls() async throws {
        let transport = ollama([
            [
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"","thinking":"I should check the weather."},"done":false}"#,
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"getWeather","arguments":{"city":"Munich"}}}]},"done":false}"#,
                #"{"model":"m","created_at":"2026-10-16T10:00:00.000Z","message":{"role":"assistant","content":""},"done":true}"#,
            ],
            [
                #"{"model":"m","created_at":"2026-10-16T10:00:01.000Z","message":{"role":"assistant","content":"It's sunny."},"done":false}"#,
                #"{"model":"m","created_at":"2026-10-16T10:00:01.000Z","message":{"role":"assistant","content":""},"done":true}"#,
            ],
        ])
        let model = OllamaLanguageModel(
            baseURL: URL(string: "http://localhost:11434")!,
            model: "m",
            transport: transport
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])
        let response = try await session.streamResponse(to: "Weather in Munich?").collect()
        #expect(response.content == "It's sunny.")

        let second = try #require(transport.bodies.count == 2 ? transport.bodies[1] : nil)
        let messages = try #require(
            (try JSONSerialization.jsonObject(with: second) as? [String: Any])?["messages"] as? [[String: Any]]
        )
        let asked = try #require(messages.first { $0["tool_calls"] != nil })
        #expect(asked["thinking"] as? String == "I should check the weather.")
        #expect(session.transcript.contains { if case .reasoning = $0 { true } else { false } })
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
}

/// Answers each request with the next of its scripted replies, as lines of a stream, and keeps
/// what each request sent.

/// A transport whose replies are rounds of Ollama's JSON lines.
private func ollama(_ rounds: [[String]]) -> FakeTransport {
    FakeTransport(replies: rounds.map { .init(chunks: [$0.joined(separator: "\n") + "\n"]) })
}
