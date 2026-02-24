#if canImport(AsyncHTTPClient)
    import AsyncHTTPClient

    public typealias SessionType = HTTPClient

    public func makeDefaultSession() -> SessionType {
        return HTTPClient.shared
    }
#else
    import Foundation
    #if canImport(FoundationNetworking)
        import FoundationNetworking
    #endif

    public typealias SessionType = URLSession

    public func makeDefaultSession() -> SessionType {
        return URLSession(configuration: .default)
    }
#endif
