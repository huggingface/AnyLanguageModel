import Testing

@testable import AnyLanguageModel

@Suite("URLSession Extensions")
struct URLSessionExtensionsTests {
    @Test func invalidResponseDescriptionMatchesExpectedText() {
        let error = URLSessionError.invalidResponse
        #expect(error.description == "Invalid response")
    }

    @Test func httpErrorDescriptionIncludesStatusCodeAndDetail() {
        let error = URLSessionError.httpError(statusCode: 429, detail: "rate limit")
        #expect(error.description == "HTTP error (Status 429): rate limit")
    }

    @Test func decodingErrorDescriptionIncludesDetail() {
        let error = URLSessionError.decodingError(detail: "keyNotFound")
        #expect(error.description == "Decoding error: keyNotFound")
    }
}
