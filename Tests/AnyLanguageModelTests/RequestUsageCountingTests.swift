import Foundation
import Testing

@testable import AnyLanguageModel

/// How a session counts the requests of a response it doesn't stream: each as the model reports it,
/// and what the model didn't report from the response.
@Suite("Counting a response's requests")
struct RequestUsageCountingTests {
    private static let request = LanguageModelSession.Usage(
        input: .init(totalTokenCount: 100, cachedTokenCount: 0),
        output: .init(totalTokenCount: 20, reasoningTokenCount: 0)
    )

    private enum Failure: Error { case failed }

    /// A model that reports a request with `usage` as it completes, then does `then`, and answers
    /// with the same usage, as a provider does.
    private static func reporting(
        then: @escaping @Sendable () async throws -> Void = {}
    ) -> MockLanguageModel {
        var model = MockLanguageModel { _, _ in
            var requests = RequestsUsage()
            requests.add(request)
            try await then()
            return "Done"
        }
        model.usage = request
        return model
    }

    @Test func aModelThatDoesntReportIsCountedFromItsResponse() async throws {
        var model = MockLanguageModel.fixed("Done")
        model.usage = Self.request
        let session = LanguageModelSession(model: model)
        _ = try await session.respond(to: "Hi")
        #expect(session.usage == Self.request)
    }

    @Test func aModelThatReportsIsCountedOnce() async throws {
        let session = LanguageModelSession(model: Self.reporting())
        let response = try await session.respond(to: "Hi")
        #expect(response.usage == Self.request)
        #expect(session.usage == Self.request)
    }

    @Test func aResponseThatThrowsKeepsWhatItsRequestsReported() async throws {
        let session = LanguageModelSession(model: Self.reporting(then: { throw Failure.failed }))
        await #expect(throws: Failure.self) { try await session.respond(to: "Hi") }
        #expect(session.usage == Self.request)
        #expect(!session.isResponding)
    }

    @Test func aCancelledResponseKeepsWhatItsRequestsReported() async throws {
        let session = LanguageModelSession(model: Self.reporting(then: { try await Task.sleep(for: .seconds(30)) }))
        let task = Task { try await session.respond(to: "Hi") }
        while session.usage == .zero { try await Task.sleep(for: .milliseconds(5)) }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(session.usage == Self.request)
    }

    /// A model of another library that wraps a built-in one and calls it through `respond`: what
    /// the built-in model's requests used is counted for the response, once, and kept when the
    /// wrapper then throws.
    @Test func aWrappedModelsRequestsAreCountedForTheResponse() async throws {
        let answered = LanguageModelSession(model: Wrapping(inner: Self.reporting(), thenThrows: false))
        _ = try await answered.respond(to: "Hi")
        #expect(answered.usage == Self.request)

        let failed = LanguageModelSession(model: Wrapping(inner: Self.reporting(), thenThrows: true))
        await #expect(throws: Wrapping.Failure.self) { try await failed.respond(to: "Hi") }
        #expect(failed.usage == Self.request)
    }

    /// What a tool asks of a model is its own, even when it reaches a provider directly.
    @Test func whatAToolUsesIsntCountedForTheResponseThatCalledIt() async throws {
        let session = LanguageModelSession(
            model: Self.reporting(then: {
                _ = try await Lookup().makeOutputSegments(from: GeneratedContent(properties: ["term": "tea"]))
            })
        )
        _ = try await session.respond(to: "Hi")
        #expect(session.usage == Self.request)
    }

    /// A stream counts its own usage from its snapshots, so one made within a response, such as by
    /// a tool, isn't counted for that response too.
    @Test func aStreamMadeWithinAResponseIsntCountedForIt() async throws {
        let streamed = LanguageModelSession(model: Self.reporting())
        let session = LanguageModelSession(
            model: Self.reporting(then: {
                _ = try await streamed.streamResponse(to: "Inner").collect()
            })
        )
        _ = try await session.respond(to: "Hi")
        #expect(session.usage == Self.request)
    }
}

/// A tool that reaches a provider itself, reporting what its request used.
private struct Lookup: Tool {
    let name = "lookup"
    let description = "Looks something up."
    @Generable struct Arguments { var term: String }
    func call(arguments: Arguments) async throws -> String {
        var requests = RequestsUsage()
        requests.add(
            LanguageModelSession.Usage(
                input: .init(totalTokenCount: 7, cachedTokenCount: 0),
                output: .init(totalTokenCount: 3, reasoningTokenCount: 0)
            )
        )
        return "Found"
    }
}

/// A model as another library would write one: it wraps a built-in model and calls it through
/// `respond`, then, when `thenThrows`, fails after it.
private struct Wrapping: LanguageModel {
    enum Failure: Error { case afterTheWrappedModel }

    let inner: MockLanguageModel
    let thenThrows: Bool

    var availability: Availability<MockLanguageModel.UnavailableReason> { inner.availability }

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        let response = try await inner.respond(
            within: session, to: prompt, generating: type, includeSchemaInPrompt: includeSchemaInPrompt, options: options)
        if thenThrows { throw Failure.afterTheWrappedModel }
        return response
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        inner.streamResponse(
            within: session, to: prompt, generating: type, includeSchemaInPrompt: includeSchemaInPrompt, options: options)
    }
}
