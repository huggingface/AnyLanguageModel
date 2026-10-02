import Foundation

/// The error details of a `response.failed` streaming event.
///
/// The OpenAI Responses API and Open Responses send the failed response
/// with an `error` object that has a `code` and a `message`.
struct ResponseStreamFailure: Sendable, Equatable {
    var code: String?
    var message: String?

    private enum ResponseKeys: String, CodingKey { case error }
    private enum ErrorKeys: String, CodingKey { case code, message }

    /// Reads the error details from the response under `key`.
    ///
    /// Returns `nil` when the event has no response or the response has no error object.
    /// A missing or malformed `code` or `message` becomes `nil`
    /// so the failure is still reported.
    init?<Key: CodingKey>(from container: KeyedDecodingContainer<Key>, forKey key: Key) {
        guard let response = try? container.nestedContainer(keyedBy: ResponseKeys.self, forKey: key),
            let error = try? response.nestedContainer(keyedBy: ErrorKeys.self, forKey: .error)
        else {
            return nil
        }
        self.code = try? error.decodeIfPresent(String.self, forKey: .code)
        self.message = try? error.decodeIfPresent(String.self, forKey: .message)
    }
}

/// Describes a `response.failed` event for an error description.
func streamFailureDescription(code: String?, message: String?) -> String {
    switch (code, message) {
    case let (code?, message?): return "The response failed while streaming (\(code)): \(message)"
    case let (code?, nil): return "The response failed while streaming (\(code))"
    case let (nil, message?): return "The response failed while streaming: \(message)"
    case (nil, nil): return "The response failed while streaming"
    }
}
