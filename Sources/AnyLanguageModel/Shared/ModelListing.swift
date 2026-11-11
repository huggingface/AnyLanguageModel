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
