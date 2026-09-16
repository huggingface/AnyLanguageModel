#if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS)
    import Foundation
    import FoundationModels

    /// A language model that uses Apple's Private Cloud Compute.
    ///
    /// Use this model to generate text with Apple's larger server-hosted models,
    /// running on Apple silicon servers under the Private Cloud Compute privacy
    /// architecture. Requests are stateless and cryptographically attested, and
    /// no data is retained.
    ///
    /// Apps need the Private Cloud Compute entitlement to use this model.
    ///
    /// ```swift
    /// let model = PrivateCloudComputeLanguageModel.default
    /// let session = LanguageModelSession(model: model)
    /// ```
    @available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *)
    public struct PrivateCloudComputeLanguageModel: LanguageModel {
        /// The reason the model is unavailable.
        public typealias UnavailableReason = FoundationModels.PrivateCloudComputeLanguageModel.Availability
            .UnavailableReason

        let pccModel: FoundationModels.PrivateCloudComputeLanguageModel
        private let wrapped: FoundationLanguageModel<FoundationModels.PrivateCloudComputeLanguageModel>

        /// The default Private Cloud Compute language model.
        public static var `default`: PrivateCloudComputeLanguageModel {
            PrivateCloudComputeLanguageModel()
        }

        /// Creates the default Private Cloud Compute language model.
        public init() {
            let pccModel = FoundationModels.PrivateCloudComputeLanguageModel()
            self.pccModel = pccModel
            self.wrapped = FoundationLanguageModel(pccModel)
        }

        /// The current quota usage for Private Cloud Compute requests.
        public var quotaUsage: FoundationModels.PrivateCloudComputeLanguageModel.QuotaUsage {
            pccModel.quotaUsage
        }

        /// Whether the model accepts image input.
        public var supportsImageInput: Bool {
            pccModel.capabilities.contains(.vision)
        }

        /// The availability status for the Private Cloud Compute language model.
        public var availability: Availability<UnavailableReason> {
            switch pccModel.availability {
            case .available:
                .available
            case .unavailable(let reason):
                .unavailable(reason)
            }
        }

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            try await wrapped.respond(
                within: session,
                to: prompt,
                generating: type,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        public func respond(
            within session: LanguageModelSession,
            to prompt: Prompt,
            schema: GenerationSchema,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<GeneratedContent> {
            try await wrapped.respond(
                within: session,
                to: prompt,
                schema: schema,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        public func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            wrapped.streamResponse(
                within: session,
                to: prompt,
                generating: type,
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
            wrapped.streamResponse(
                within: session,
                to: prompt,
                schema: schema,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        public func logFeedbackAttachment(
            within session: LanguageModelSession,
            sentiment: LanguageModelFeedback.Sentiment?,
            issues: [LanguageModelFeedback.Issue],
            desiredOutput: Transcript.Entry?
        ) -> Data {
            let fmSession = FoundationModels.LanguageModelSession(
                model: pccModel,
                tools: session.tools.toFoundationModels(),
                instructions: session.instructions?.toFoundationModels()
            )
            return fmSession.logFeedbackAttachment(
                sentiment: sentiment?.toFoundationModels(),
                issues: issues.map { $0.toFoundationModels() },
                desiredOutput: nil
            )
        }
    }
#endif
