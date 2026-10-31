import Foundation
import HTTPTypes
import Testing

// Exercise the public error surface through each provider's actual stream decoder.
import AnyLanguageModel

@Suite("Provider stream errors")
struct ProviderStreamErrorTests {
    enum Provider: CaseIterable, Sendable {
        case anthropic, chatCompletions, responses, openResponses, gemini

        var partial: String {
            switch self {
            case .anthropic:
                return
                    #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Partial answer"}}"#
            case .chatCompletions:
                return #"{"id":"fixture","choices":[{"delta":{"content":"Partial answer"}}]}"#
            case .responses, .openResponses:
                return #"{"type":"response.output_text.delta","delta":"Partial answer","sequence_number":0}"#
            case .gemini:
                return #"{"candidates":[{"content":{"role":"model","parts":[{"text":"Partial answer"}]}}]}"#
            }
        }

        var failure: String {
            switch self {
            case .anthropic:
                return #"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#
            case .chatCompletions:
                return #"{"error":{"type":"server_error","code":"fixture_error","message":"Fixture failure"}}"#
            case .responses, .openResponses:
                return
                    #"{"type":"error","code":"fixture_error","message":"Fixture failure","param":null,"sequence_number":1}"#
            case .gemini:
                return #"{"error":{"code":500,"status":"INTERNAL","message":"Fixture failure"}}"#
            }
        }

        func model(transport: any HTTPTransport) -> any LanguageModel {
            switch self {
            case .anthropic:
                return AnthropicLanguageModel(apiKey: "fixture", model: "fixture", transport: transport)
            case .chatCompletions:
                return OpenAILanguageModel(
                    apiKey: "fixture",
                    model: "fixture",
                    apiVariant: .chatCompletions,
                    transport: transport
                )
            case .responses:
                return OpenAILanguageModel(
                    apiKey: "fixture",
                    model: "fixture",
                    apiVariant: .responses,
                    transport: transport
                )
            case .openResponses:
                return OpenResponsesLanguageModel(
                    baseURL: URL(string: "https://fixture.invalid")!,
                    apiKey: "fixture",
                    model: "fixture",
                    transport: transport
                )
            case .gemini:
                return GeminiLanguageModel(apiKey: "fixture", model: "fixture", transport: transport)
            }
        }
    }

    @Test(arguments: Provider.allCases, [false, true])
    func knownFailureTerminatesTheStream(provider: Provider, includePartial: Bool) async throws {
        let events = (includePartial ? [provider.partial] : []) + [provider.failure]
        let body = events.map { "data: \($0)\n\n" }.joined()
        let requestID = HTTPField.Name("X-Request-ID")!
        // Split the failure across transport chunks as well as SSE events.
        let middle = body.index(body.startIndex, offsetBy: body.count / 2)
        let transport = FakeTransport(replies: [
            .init(
                headerFields: [requestID: "fixture-request"],
                chunks: [String(body[..<middle]), String(body[middle...])]
            )
        ])
        let session = LanguageModelSession(model: provider.model(transport: transport))
        var snapshots: [String] = []
        let error = try await #require(throws: HTTPTransportError.self) {
            for try await snapshot in session.streamResponse(to: "Question") {
                snapshots.append(snapshot.content)
            }
        }
        #expect(snapshots == (includePartial ? ["Partial answer"] : []))
        #expect(error.kind == .providerError)
        #expect(error.response?.status == .ok)
        #expect(error.response?.headerFields[requestID] == "fixture-request")
        #expect(error.body == Data(provider.failure.utf8))
        #expect(error.underlyingError == nil)
        #expect(error.description.contains(provider.failure))
    }

    @Test(arguments: Provider.allCases)
    func informationalAndUnknownEventsStillAllowNormalCompletion(provider: Provider) async throws {
        let events = [
            #"{"type":"ping"}"#,
            #"{"type":"future.event","message":"informational"}"#,
            #"{"type":"future.event","error":{"message":"metadata"}}"#,
            #"{"error":null}"#,
            #"{"error":{"type":"metadata"}}"#,
            "not-json",
            provider.partial,
            "[DONE]",
        ]
        let body = events.map { "data: \($0)\n\n" }.joined()
        let transport = FakeTransport(replies: [.init(chunks: [body])])
        let session = LanguageModelSession(model: provider.model(transport: transport))
        let response = try await session.streamResponse(to: "Question").collect()
        #expect(response.content == "Partial answer")
    }
}
