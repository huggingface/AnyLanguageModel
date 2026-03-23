#if canImport(AsyncHTTPClient)
    import AsyncHTTPClient

    public typealias HTTPSession = HTTPClient

    public func makeDefaultSession() -> HTTPSession {
        return HTTPClient.shared
    }
#else
    import Foundation
    #if canImport(FoundationNetworking)
        import FoundationNetworking
    #endif

    public typealias HTTPSession = URLSession

    public func makeDefaultSession() -> HTTPSession {
        return URLSession(configuration: .default)
    }
#endif
