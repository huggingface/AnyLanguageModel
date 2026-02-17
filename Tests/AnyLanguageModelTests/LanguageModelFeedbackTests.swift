import Testing

@testable import AnyLanguageModel

@Suite("LanguageModelFeedback")
struct LanguageModelFeedbackTests {
    @Test func sentimentExposesAllCases() {
        #expect(LanguageModelFeedback.Sentiment.allCases.count == 3)
        #expect(LanguageModelFeedback.Sentiment.allCases.contains(.positive))
        #expect(LanguageModelFeedback.Sentiment.allCases.contains(.negative))
        #expect(LanguageModelFeedback.Sentiment.allCases.contains(.neutral))
    }

    @Test func issueCategoryExposesAllCases() {
        #expect(LanguageModelFeedback.Issue.Category.allCases.count == 8)
        #expect(LanguageModelFeedback.Issue.Category.allCases.contains(.unhelpful))
        #expect(LanguageModelFeedback.Issue.Category.allCases.contains(.tooVerbose))
        #expect(LanguageModelFeedback.Issue.Category.allCases.contains(.didNotFollowInstructions))
        #expect(LanguageModelFeedback.Issue.Category.allCases.contains(.incorrect))
        #expect(LanguageModelFeedback.Issue.Category.allCases.contains(.stereotypeOrBias))
        #expect(LanguageModelFeedback.Issue.Category.allCases.contains(.suggestiveOrSexual))
        #expect(LanguageModelFeedback.Issue.Category.allCases.contains(.vulgarOrOffensive))
        #expect(LanguageModelFeedback.Issue.Category.allCases.contains(.triggeredGuardrailUnexpectedly))
    }

    @Test func issueInitializerStoresCategoryAndExplanation() {
        let issue = LanguageModelFeedback.Issue(
            category: .tooVerbose,
            explanation: "Response includes extra paragraphs."
        )

        #expect(issue.category == .tooVerbose)
        #expect(issue.explanation == "Response includes extra paragraphs.")
    }

    @Test func feedbackInitializerStoresSentimentAndIssues() {
        let issue = LanguageModelFeedback.Issue(category: .incorrect, explanation: nil)
        let feedback = LanguageModelFeedback(sentiment: .negative, issues: [issue])

        #expect(feedback.sentiment == .negative)
        #expect(feedback.issues.count == 1)
        #expect(feedback.issues.first?.category == .incorrect)
        #expect(feedback.issues.first?.explanation == nil)
    }
}
