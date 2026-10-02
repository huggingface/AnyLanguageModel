import Foundation
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

    @Test func desiredResponseTextIsLoggedAsTextResponse() throws {
        let session = LanguageModelSession(model: FeedbackRecordingModel())
        let data = session.logFeedbackAttachment(
            sentiment: .negative,
            issues: [.init(category: .incorrect)],
            desiredResponseText: "Paris"
        )

        let entry = try JSONDecoder().decode(Transcript.Entry.self, from: data)
        guard case .response(let response) = entry,
            case .text(let text)? = response.segments.first
        else {
            Issue.record("Expected a response entry with a text segment, got \(entry)")
            return
        }
        #expect(response.segments.count == 1)
        #expect(text.content == "Paris")
    }

    @Test func desiredResponseContentIsLoggedAsStructuredResponse() throws {
        let session = LanguageModelSession(model: FeedbackRecordingModel())
        let content = GeneratedContent(properties: ["city": "Paris"])
        let data = session.logFeedbackAttachment(
            sentiment: .negative,
            desiredResponseContent: content
        )

        let entry = try JSONDecoder().decode(Transcript.Entry.self, from: data)
        guard case .response(let response) = entry,
            case .structure(let structure)? = response.segments.first
        else {
            Issue.record("Expected a response entry with a structured segment, got \(entry)")
            return
        }
        #expect(structure.source == "GeneratedContent")
        #expect(structure.content == content)
    }

    @Test func nilDesiredResponseLogsNoOutput() {
        let session = LanguageModelSession(model: FeedbackRecordingModel())
        #expect(session.logFeedbackAttachment(sentiment: .positive, desiredResponseText: nil).isEmpty)
        #expect(session.logFeedbackAttachment(sentiment: .positive, desiredResponseContent: nil).isEmpty)
    }
}

/// A model that returns the desired output it receives, encoded as JSON.
private struct FeedbackRecordingModel: LanguageModel {
    typealias UnavailableReason = Never

    func respond<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
        fatalError("Not used")
    }

    func streamResponse<Content>(
        within session: LanguageModelSession,
        to prompt: Prompt,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
        fatalError("Not used")
    }

    func logFeedbackAttachment(
        within session: LanguageModelSession,
        sentiment: LanguageModelFeedback.Sentiment?,
        issues: [LanguageModelFeedback.Issue],
        desiredOutput: Transcript.Entry?
    ) -> Data {
        guard let desiredOutput else { return Data() }
        return (try? JSONEncoder().encode(desiredOutput)) ?? Data()
    }
}
