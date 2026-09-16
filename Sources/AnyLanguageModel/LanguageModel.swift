import Foundation

public protocol LanguageModel: Sendable {
    associatedtype UnavailableReason

    /// The type of custom generation options this model accepts.
    ///
    /// Models can define their own custom options types with extended properties
    /// by setting this to a custom type conforming to ``CustomGenerationOptions``.
    /// The default is `Never`, indicating no custom options are supported.
    associatedtype CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions = Never

    var availability: Availability<UnavailableReason> { get }

    func prewarm(
        for session: LanguageModelSession,
        promptPrefix: Prompt?
    )

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable

    /// Generates content using the supplied schema.
    ///
    /// Providers that only implement the generic requirements use their existing
    /// `GeneratedContent` behavior until they implement this requirement.
    func respond(
        within session: LanguageModelSession,
        to prompt: Prompt,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<GeneratedContent>

    /// Streams content using the supplied schema.
    ///
    /// Providers that only implement the generic requirements use their existing
    /// `GeneratedContent` behavior until they implement this requirement.
    func streamResponse(
        within session: LanguageModelSession,
        to prompt: Prompt,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<GeneratedContent>

    func logFeedbackAttachment(
        within session: LanguageModelSession,
        sentiment: LanguageModelFeedback.Sentiment?,
        issues: [LanguageModelFeedback.Issue],
        desiredOutput: Transcript.Entry?
    ) -> Data
}

// MARK: - Default Implementation

extension LanguageModel {
    public func respond(
        within session: LanguageModelSession,
        to prompt: Prompt,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<GeneratedContent> {
        try await respond(
            within: session,
            to: prompt,
            generating: GeneratedContent.self,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    public func streamResponse(
        within session: LanguageModelSession,
        to prompt: Prompt,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<GeneratedContent> {
        streamResponse(
            within: session,
            to: prompt,
            generating: GeneratedContent.self,
            includeSchemaInPrompt: includeSchemaInPrompt,
            options: options
        )
    }

    public var isAvailable: Bool {
        if case .available = availability {
            return true
        } else {
            return false
        }
    }

    public func prewarm(
        for session: LanguageModelSession,
        promptPrefix: Prompt? = nil
    ) {
        return
    }

    public func logFeedbackAttachment(
        within session: LanguageModelSession,
        sentiment: LanguageModelFeedback.Sentiment? = nil,
        issues: [LanguageModelFeedback.Issue] = [],
        desiredOutput: Transcript.Entry? = nil
    ) -> Data {
        return Data()
    }
}

extension LanguageModel where UnavailableReason == Never {
    public var availability: Availability<UnavailableReason> {
        return .available
    }
}
