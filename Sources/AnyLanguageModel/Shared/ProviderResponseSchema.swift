/// A response schema in the form that a provider accepts.
///
/// Some providers need an object at the root of a response schema.
/// OpenAI's structured outputs, for example,
/// reject a root that isn't an object or that uses `anyOf`.
/// For those providers, a schema with another root,
/// such as an enumeration whose cases have associated values or an array,
/// becomes the one required property of an object,
/// and the response is unwrapped before it's decoded.
///
/// Wrapping happens only here, at the provider boundary,
/// so `GenerationSchema` keeps the form that Foundation Models gives it.
struct ProviderResponseSchema {
    /// The roots that a provider doesn't accept.
    enum Wrapping {
        /// Any root that isn't an object.
        case nonObjectRoots
        /// A root that's a union of schemas.
        case unionRoots
    }

    /// The name of the property that holds a wrapped response.
    static let wrapperProperty = "response"

    /// The schema to send to the provider.
    let schema: GenerationSchema

    /// The property that holds the response, if the schema wraps it.
    let wrapperProperty: String?

    /// Creates the response schema to send for content of the given type.
    ///
    /// String content sends no schema, so it's never wrapped.
    init<Content: Generable>(_ schema: GenerationSchema, generating type: Content.Type, wrapping: Wrapping) {
        let wraps: Bool
        switch (wrapping, schema.resolvedRoot) {
        case (_, .object?):
            wraps = false
        case (.nonObjectRoots, _), (.unionRoots, .anyOf?):
            wraps = true
        case (.unionRoots, _):
            wraps = false
        }
        guard wraps, type != String.self else {
            self.schema = schema
            self.wrapperProperty = nil
            return
        }
        self.schema = schema.wrappingRoot(inProperty: Self.wrapperProperty)
        self.wrapperProperty = Self.wrapperProperty
    }

    /// Returns the response content in JSON text, unwrapped if the schema wraps it.
    func content(json text: String) throws -> GeneratedContent {
        Self.unwrapping(try GeneratedContent(json: text), fromProperty: wrapperProperty)
    }

    /// Returns the value of the wrapper property
    /// if the content is an object with only that property.
    ///
    /// Other content is returned as the provider sent it,
    /// so a provider that doesn't apply the response schema can return content without the wrapper.
    static func unwrapping(_ content: GeneratedContent, fromProperty property: String?) -> GeneratedContent {
        guard let property,
            case .structure(let properties, _) = content.kind,
            properties.count == 1,
            let wrapped = properties[property]
        else { return content }
        return wrapped
    }
}

extension GenerationSchema {
    /// The root, after following references.
    var resolvedRoot: Node? {
        resolving(root)
    }

    /// Returns the schema with its root as the one required property of an object.
    func wrappingRoot(inProperty property: String) -> GenerationSchema {
        let object = ObjectNode(
            description: nil,
            properties: [property: root],
            required: [property],
            propertyOrder: [property]
        )
        return GenerationSchema(root: .object(object), defs: defs)
    }
}
