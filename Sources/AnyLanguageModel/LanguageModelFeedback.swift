/// Feedback about a model's response.
///
/// - Note: The ``sentiment`` and ``issues`` properties are exclusive to AnyLanguageModel.
///   They're public for language models outside this module;
///   Foundation Models doesn't make them public.
public struct LanguageModelFeedback {
    /// A sentiment regarding the model's response.
    public enum Sentiment: Sendable, CaseIterable, Equatable, Hashable {
        /// A positive sentiment
        case positive

        /// A negative sentiment
        case negative

        /// A neutral sentiment
        case neutral
    }

    /// A sentiment regarding the model's response.
    public let sentiment: Sentiment

    /// An issue with the model's response.
    public struct Issue: Sendable {
        /// Categories for model response issues.
        public enum Category: Sendable, CaseIterable, Equatable, Hashable {

            /// The response was not unhelpful.
            ///
            /// An unhelpful issue might be where you asked for a recipe, and the model gave you a list of
            /// ingredients but not amounts.
            case unhelpful

            /// The response was too verbose.
            ///
            /// A verbose issue might be where you asked for a simple recipe, and the model wrote introductory
            /// and conclusion paragraphs.
            case tooVerbose

            /// The model did not follow instructions correctly.
            ///
            /// An instruction issue might be where you asked for a recipe in numbered steps, and the model
            /// provided a recipe but didn't number the steps.
            case didNotFollowInstructions

            /// The model provided an incorrect response.
            ///
            /// An incorrect issue might be where you asked how to make a pizza, and the model suggested using glue.
            case incorrect

            /// The model exhibited bias or perpetuated a sterotype.
            ///
            /// A stereotype or bias issue might be where you ask the model to summarize an article written by
            /// a male, and the model doesn't state the authors sex, but the model uses male pronouns.
            case stereotypeOrBias

            /// The model produces suggestive or sexual material.
            ///
            /// A suggestive or sexual issue might be where you ask the model to draft a script for a school
            /// play, and it includes a sex scene.
            case suggestiveOrSexual

            /// The model produces vulgar or offensive material.
            ///
            /// A vulgar or offensive issue might be where you ask the model to draft a complaint about poor
            /// customer service, and it uses profanity.
            case vulgarOrOffensive

            /// The model throws a guardrail violation when it shouldn't.
            ///
            /// An unexpected guardrail issue might be where you ask for a cake recipe, and the framework
            /// throws a guardrail violation error.
            case triggeredGuardrailUnexpectedly
        }

        /// The category of the issue.
        public let category: Category

        /// The explanation of the issue.
        public let explanation: String?

        /// Creates a new issue
        ///
        /// - Parameters:
        ///   - category: A category for this issue.
        ///   - explanation: An optional explanation of this issue.
        public init(category: Category, explanation: String? = nil) {
            self.category = category
            self.explanation = explanation
        }
    }

    /// Issues with the model's response.
    public let issues: [Issue]

    /// Creates a new language model feedback object.
    ///
    /// - Parameters:
    ///   - sentiment: A sentiment for the model's response.
    ///   - issues: Issues with the model's response.
    ///
    /// - Note: This initializer is exclusive to AnyLanguageModel.
    ///   It's public so that language models outside this module
    ///   can create feedback;
    ///   Foundation Models doesn't make it public.
    public init(sentiment: Sentiment, issues: [Issue]) {
        self.sentiment = sentiment
        self.issues = issues
    }
}
