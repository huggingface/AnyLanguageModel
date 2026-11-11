import Foundation
import HTTPTypes
import Testing

@testable import AnyLanguageModel

/// What a response's request tells the provider about the caller: Anthropic its key, its API
/// version and its betas; Gemini its key.
@Suite("Provider request headers")
struct ProviderRequestHeadersTests {
    @Test func anthropicSendsItsKeyVersionAndBetas() async throws {
        let transport = FakeTransport(replies: [.init(chunks: [Self.anthropicStream])])
        let model = AnthropicLanguageModel(
            apiKey: "secret",
            apiVersion: "2023-06-01",
            betas: ["first", "second"],
            model: "m",
            transport: transport
        )

        _ = try await LanguageModelSession(model: model).streamResponse(to: "Question").collect()

        let headers = try #require(transport.requests.first?.headerFields)
        #expect(headers[HTTPField.Name("x-api-key")!] == "secret")
        #expect(headers[HTTPField.Name("anthropic-version")!] == "2023-06-01")
        #expect(headers[HTTPField.Name("anthropic-beta")!] == "first,second")
    }

    @Test func geminiSendsItsKey() async throws {
        let transport = FakeTransport(replies: [.init(chunks: [Self.geminiStream])])
        let model = GeminiLanguageModel(apiKey: "secret", model: "m", transport: transport)

        _ = try await LanguageModelSession(model: model).streamResponse(to: "Question").collect()

        let headers = try #require(transport.requests.first?.headerFields)
        #expect(headers[HTTPField.Name("x-goog-api-key")!] == "secret")
    }

    private static let anthropicStream = [
        #"{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"m","usage":{"input_tokens":1,"output_tokens":0}}}"#,
        #"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Answer"}}"#,
        #"{"type":"content_block_stop","index":0}"#,
        #"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}"#,
        #"{"type":"message_stop"}"#,
    ].map { "data: \($0)\n\n" }.joined()

    private static let geminiStream =
        "data: "
        + #"{"candidates":[{"content":{"role":"model","parts":[{"text":"Answer"}]},"finishReason":"STOP"}]}"#
        + "\n\n"
}
