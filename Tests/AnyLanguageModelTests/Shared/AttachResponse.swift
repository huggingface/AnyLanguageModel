import Testing

@testable import AnyLanguageModel

/// Attaches a generated response to the running test,
/// so it can be read in the test report after a run.
///
/// Text responses are attached as plain text.
/// Structured responses are attached as the JSON the model produced.
/// The attachment is named after the current test,
/// followed by `label` when a test attaches more than one response.
func attachResponse<Content: Generable>(
    _ response: LanguageModelSession.Response<Content>,
    _ label: String? = nil,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let name = attachmentName(label)
    if let text = response.content as? String {
        Attachment.record(text, named: "\(name).txt", sourceLocation: sourceLocation)
    } else {
        Attachment.record(response.rawContent.jsonString, named: "\(name).json", sourceLocation: sourceLocation)
    }
}

private func attachmentName(_ label: String?) -> String {
    var name = Test.current?.name ?? "response"
    if name.hasSuffix("()") {
        name.removeLast(2)
    }
    return label.map { "\(name) \($0)" } ?? name
}
