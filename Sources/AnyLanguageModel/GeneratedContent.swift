import Foundation
import CoreFoundation

/// A type that represents structured, generated content.
///
/// Generated content may contain a single value, an array, or key-value pairs with unique keys.
public struct GeneratedContent: Sendable, Equatable, Generable, CustomDebugStringConvertible, Codable {
    /// An instance of the generation schema.
    public static var generationSchema: GenerationSchema {
        // GeneratedContent is self-describing, it doesn't have a fixed schema
        // This is a placeholder that should rarely be called
        GenerationSchema.primitive(
            GeneratedContent.self,
            node: .string(
                GenerationSchema.StringNode(description: "Dynamic generated content", pattern: nil, enumChoices: nil)
            )
        )
    }

    /// A unique id that is stable for the duration of a generated response.
    ///
    /// A ``LanguageModelSession`` produces instances of `GeneratedContent` that have a
    /// non-nil `id`. When you stream a response, the `id` is the same for all partial generations in the
    /// response stream.
    ///
    /// Instances of `GeneratedContent` that you produce manually with initializers have a nil `id`
    /// because the framework didn't create them as part of a generation.
    public var id: GenerationID?

    /// The kind representation of this generated content.
    ///
    /// This property provides access to the content in a strongly-typed enum representation,
    /// preserving the hierarchical structure of the data and the generation IDs.
    public let kind: Kind

    /// Creates generated content from another value.
    ///
    /// This is used to satisfy `Generable.init(_:)`.
    public init(_ content: GeneratedContent) throws {
        self = content
    }

    /// A representation of this instance.
    public var generatedContent: GeneratedContent { self }

    /// Creates generated content representing a structure with the properties you specify.
    ///
    /// The order of properties is important. For ``Generable`` types, the order
    /// must match the order properties in the types `schema`.
    public init(
        properties: KeyValuePairs<String, any ConvertibleToGeneratedContent>,
        id: GenerationID? = nil
    ) {
        var dict: [String: GeneratedContent] = [:]
        var keys: [String] = []
        for (key, value) in properties {
            dict[key] = value.generatedContent
            keys.append(key)
        }
        self.init(kind: .structure(properties: dict, orderedKeys: keys), id: id)
    }

    /// Creates new generated content from the key-value pairs in the given sequence,
    /// using a combining closure to determine the value for any duplicate keys.
    ///
    /// The order of properties is important. For ``Generable`` types, the order
    /// must match the order properties in the types `schema`.
    ///
    /// You use this initializer to create generated content when you have a sequence
    /// of key-value tuples that might have duplicate keys. As the content is
    /// built, the initializer calls the `combine` closure with the current and
    /// new values for any duplicate keys. Pass a closure as `combine` that
    /// returns the value to use in the resulting content: The closure can
    /// choose between the two values, combine them to produce a new value, or
    /// even throw an error.
    ///
    /// The following example shows how to choose the first and last values for
    /// any duplicate keys:
    ///
    /// ```swift
    ///     let content = GeneratedContent(
    ///       properties: [("name", "John"), ("name", "Jane"), ("married": true)],
    ///       uniquingKeysWith: { (first, _ in first }
    ///     )
    ///     // GeneratedContent(["name": "John", "married": true])
    /// ```
    ///
    /// - Parameters:
    ///   - properties: A sequence of key-value pairs to use for the new content.
    ///   - id: A unique id associated with GeneratedContent.
    ///   - uniquingKeysWith: A closure that is called with the values for any duplicate
    ///     keys that are encountered. The closure returns the desired value for
    ///     the final content.
    public init<S>(
        properties: S,
        id: GenerationID? = nil,
        uniquingKeysWith combine: (GeneratedContent, GeneratedContent) throws ->
            some ConvertibleToGeneratedContent
    ) rethrows where S: Sequence, S.Element == (String, any ConvertibleToGeneratedContent) {
        var dict: [String: GeneratedContent] = [:]
        var keys: [String] = []

        for (key, value) in properties {
            let newContent = value.generatedContent
            if let existing = dict[key] {
                dict[key] = try combine(existing, newContent).generatedContent
            } else {
                dict[key] = newContent
                keys.append(key)
            }
        }

        self.init(kind: .structure(properties: dict, orderedKeys: keys), id: id)
    }

    /// Creates content representing an array of elements you specify.
    public init<S>(
        elements: S,
        id: GenerationID? = nil
    ) where S: Sequence, S.Element == any ConvertibleToGeneratedContent {
        let contentArray = elements.map { $0.generatedContent }
        self.init(kind: .array(contentArray), id: id)
    }

    /// Creates content that contains a single value.
    ///
    /// - Parameters:
    ///   - value: The underlying value.
    public init(_ value: some ConvertibleToGeneratedContent) {
        self = value.generatedContent
    }

    /// Creates content that contains a single value with a custom generation ID.
    ///
    /// - Parameters:
    ///   - value: The underlying value.
    ///   - id: The generation ID for this content.
    public init(_ value: some ConvertibleToGeneratedContent, id: GenerationID) {
        self.init(kind: value.generatedContent.kind, id: id)
    }

    /// Creates equivalent content from a JSON string.
    ///
    /// The JSON string you provide may be incomplete. This is useful for correctly handling partially generated responses.
    ///
    /// ```swift
    /// @Generable struct NovelIdea {
    ///   let title: String
    /// }
    ///
    /// let partial = #"{"title": "A story of"#
    /// let content = try GeneratedContent(json: partial)
    /// let idea = try NovelIdea(content)
    /// print(idea.title) // A story of
    /// ```
    public init(json: String) throws {
        // Parse JSON with support for incomplete JSON
        guard let data = json.data(using: .utf8) else {
            throw GeneratedContentError.typeMismatch
        }

        // Try to parse as complete JSON first
        if let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) {
            self = try Self.fromJSONValue(parsed)
            return
        }

        // Handle incomplete JSON by attempting to complete it
        let completedJSON = json.trimmingCharacters(in: .whitespacesAndNewlines)

        // Try adding closing braces/brackets to make it valid
        var attempts: [String] = [completedJSON]

        // If it looks like an incomplete object, try closing it
        if completedJSON.hasPrefix("{") && !completedJSON.hasSuffix("}") {
            attempts.append(completedJSON + "}")
            attempts.append(completedJSON + "\"\"}")  // incomplete string value
        }

        // If it looks like an incomplete array, try closing it
        if completedJSON.hasPrefix("[") && !completedJSON.hasSuffix("]") {
            attempts.append(completedJSON + "]")
        }

        for attempt in attempts {
            if let data = attempt.data(using: .utf8),
                let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
            {
                self = try Self.fromJSONValue(parsed)
                return
            }
        }

        // If all else fails, treat it as a string
        self.init(kind: .string(completedJSON))
    }

    private static func fromJSONValue(_ value: Any) throws -> GeneratedContent {
        if let dict = value as? [String: Any] {
            var properties: [String: GeneratedContent] = [:]
            var keys: [String] = []
            for (key, val) in dict {
                properties[key] = try fromJSONValue(val)
                keys.append(key)
            }
            return GeneratedContent(kind: .structure(properties: properties, orderedKeys: keys))
        } else if let array = value as? [Any] {
            let contents = try array.map { try fromJSONValue($0) }
            return GeneratedContent(kind: .array(contents))
        } else if let string = value as? String {
            return GeneratedContent(kind: .string(string))
        } else if let number = value as? NSNumber {
            // Check if it's a boolean
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return GeneratedContent(kind: .bool(number.boolValue))
            }
            return GeneratedContent(kind: .number(number.doubleValue))
        } else if value is NSNull {
            return GeneratedContent(kind: .null)
        } else {
            throw GeneratedContentError.typeMismatch
        }
    }

    /// Returns a JSON string representation of the generated content.
    ///
    /// ## Examples
    ///
    /// ```swift
    /// // Object with properties
    /// let content = GeneratedContent(properties: [
    ///     "name": "Johnny Appleseed",
    ///     "age": 30,
    /// ])
    /// print(content.jsonString)
    /// // Output: {"name": "Johnny Appleseed", "age": 30}
    /// ```
    public var jsonString: String {
        do {
            let jsonValue = try toJSONValue()
            let data = try JSONSerialization.data(withJSONObject: jsonValue, options: [.fragmentsAllowed])
            return String(data: data, encoding: .utf8) ?? "{}"
        } catch {
            return "{}"
        }
    }

    private func toJSONValue() throws -> Any {
        switch kind {
        case .null:
            return NSNull()
        case .bool(let value):
            return value
        case .number(let value):
            return value
        case .string(let value):
            return value
        case .array(let elements):
            return try elements.map { try $0.toJSONValue() }
        case .structure(let properties, let orderedKeys):
            var dict: [String: Any] = [:]
            for key in orderedKeys {
                if let value = properties[key] {
                    dict[key] = try value.toJSONValue()
                }
            }
            return dict
        }
    }

    /// Reads a top level, concrete partially generable type.
    public func value<Value>(_ type: Value.Type = Value.self) throws -> Value
    where Value: ConvertibleFromGeneratedContent {
        try Value(self)
    }

    /// Reads a concrete generable type from named property.
    public func value<Value>(
        _ type: Value.Type = Value.self,
        forProperty property: String
    ) throws -> Value where Value: ConvertibleFromGeneratedContent {
        guard case .structure(let properties, _) = kind,
            let value = properties[property]
        else {
            throw GeneratedContentError.propertyNotFound(property)
        }
        return try Value(value)
    }

    /// Reads an optional, concrete generable type from named property.
    public func value<Value>(
        _ type: Value?.Type = Value?.self,
        forProperty property: String
    ) throws -> Value? where Value: ConvertibleFromGeneratedContent {
        guard case .structure(let properties, _) = kind else {
            return nil
        }
        guard let value = properties[property] else {
            return nil
        }
        return try Value(value)
    }

    /// A string representation for the debug description.
    public var debugDescription: String {
        "GeneratedContent(\(kind))"
    }

    /// A Boolean that indicates whether the generated content is completed.
    public var isComplete: Bool {
        // Check if the content is structurally complete
        switch kind {
        case .null, .bool, .number, .string:
            return true
        case .array(let elements):
            return elements.allSatisfy { $0.isComplete }
        case .structure(let properties, _):
            return properties.values.allSatisfy { $0.isComplete }
        }
    }

    public static func == (a: GeneratedContent, b: GeneratedContent) -> Bool {
        a.kind == b.kind && a.id == b.id
    }
}

// MARK: - GeneratedContent.Kind

extension GeneratedContent {
    /// A representation of the different types of content that can be stored in `GeneratedContent`.
    ///
    /// `Kind` represents the various types of JSON-compatible data that can be held within
    /// a `GeneratedContent` instance, including primitive types, arrays, and structured objects.
    public enum Kind: Equatable, Sendable {

        /// Represents a null value.
        case null

        /// Represents a boolean value.
        case bool(Bool)

        /// Represents a numeric value.
        case number(Double)

        /// Represents a string value.
        case string(String)

        /// Represents an array of `GeneratedContent` elements.
        case array([GeneratedContent])

        /// Represents a structured object with key-value pairs.
        case structure(properties: [String: GeneratedContent], orderedKeys: [String])

        public static func == (a: GeneratedContent.Kind, b: GeneratedContent.Kind) -> Bool {
            switch (a, b) {
            case (.null, .null):
                return true
            case (.bool(let lhs), .bool(let rhs)):
                return lhs == rhs
            case (.number(let lhs), .number(let rhs)):
                return lhs == rhs
            case (.string(let lhs), .string(let rhs)):
                return lhs == rhs
            case (.array(let lhs), .array(let rhs)):
                return lhs == rhs
            case (.structure(let lhsProps, let lhsKeys), .structure(let rhsProps, let rhsKeys)):
                return lhsProps == rhsProps && lhsKeys == rhsKeys
            default:
                return false
            }
        }
    }

    /// Creates a new `GeneratedContent` instance with the specified kind and generation ID.
    ///
    /// This initializer provides a convenient way to create content from its kind representation.
    ///
    /// - Parameters:
    ///   - kind: The kind of content to create.
    ///   - id: An optional generation ID to associate with this content.
    public init(kind: GeneratedContent.Kind, id: GenerationID? = nil) {
        self.kind = kind
        self.id = id
    }
}

// MARK: - GeneratedContentError

public enum GeneratedContentError: Error, Hashable {
    case propertyNotFound(String)
    case typeMismatch
    case neverCannotBeInstantiated
}

// MARK: - Codable

extension GeneratedContent {
    private enum CodingKeys: String, CodingKey {
        case id
        case kind
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decodeIfPresent(GenerationID.self, forKey: .id)
        self.kind = try container.decode(Kind.self, forKey: .kind)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(id, forKey: .id)
        try container.encode(kind, forKey: .kind)
    }
}

extension GeneratedContent.Kind: Codable {
    private enum CodingKeys: String, CodingKey {
        case type
        case value
        case properties
        case orderedKeys
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "null":
            self = .null
        case "bool":
            self = .bool(try container.decode(Bool.self, forKey: .value))
        case "number":
            self = .number(try container.decode(Double.self, forKey: .value))
        case "string":
            self = .string(try container.decode(String.self, forKey: .value))
        case "array":
            self = .array(try container.decode([GeneratedContent].self, forKey: .value))
        case "structure":
            let properties = try container.decode([String: GeneratedContent].self, forKey: .properties)
            let orderedKeys = try container.decode([String].self, forKey: .orderedKeys)
            self = .structure(properties: properties, orderedKeys: orderedKeys)
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unknown kind type: \(type)"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .null:
            try container.encode("null", forKey: .type)
        case .bool(let value):
            try container.encode("bool", forKey: .type)
            try container.encode(value, forKey: .value)
        case .number(let value):
            try container.encode("number", forKey: .type)
            try container.encode(value, forKey: .value)
        case .string(let value):
            try container.encode("string", forKey: .type)
            try container.encode(value, forKey: .value)
        case .array(let elements):
            try container.encode("array", forKey: .type)
            try container.encode(elements, forKey: .value)
        case .structure(let properties, let orderedKeys):
            try container.encode("structure", forKey: .type)
            try container.encode(properties, forKey: .properties)
            try container.encode(orderedKeys, forKey: .orderedKeys)
        }
    }
}
