import Foundation
import HTTPTypes

/// What asking a provider for its models is made of: a page's address, with its query.
extension URL {
    /// This address asking `query`: a request that can't be made of them, as one that can't be sent.
    func asking(_ query: [URLQueryItem]) throws -> URL {
        guard var components = URLComponents(url: self, resolvingAgainstBaseURL: false) else {
            throw HTTPTransportError(kind: .invalidRequest)
        }
        components.queryItems = (components.queryItems ?? []) + query
        guard let url = components.url else { throw HTTPTransportError(kind: .invalidRequest) }
        return url
    }
}

/// What `body` asks a provider, with a failure read as what it says: a key missing or refused, a
/// provider that can't be reached, or what it answered instead. A cancelled request throws
/// `CancellationError`.
func askingProvider<Answer>(
    keyGiven: Bool,
    refusesKey: (HTTPTransportError) -> Bool = { _ in false },
    _ body: () async throws -> Answer
) async throws -> Answer {
    do {
        return try await body()
    } catch {
        throw try ModelUnavailability(asking: error, keyGiven: keyGiven, missingModel: false, refusesKey: refusesKey)
    }
}
