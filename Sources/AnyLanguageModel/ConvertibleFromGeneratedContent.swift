/// A type that can be initialized from generated content.
public protocol ConvertibleFromGeneratedContent: SendableMetatype {
    /// Creates an instance with the content.
    init(_ content: GeneratedContent) throws
}

// MARK: - Standard Library Extensions

extension Array: ConvertibleFromGeneratedContent & SendableMetatype
where Element: ConvertibleFromGeneratedContent {
    /// Creates an instance with the content.
    public init(_ content: GeneratedContent) throws {
        guard case .array(let elements) = content.kind else {
            throw GeneratedContentError.typeMismatch
        }
        self = try elements.map { try Element($0) }
    }
}
