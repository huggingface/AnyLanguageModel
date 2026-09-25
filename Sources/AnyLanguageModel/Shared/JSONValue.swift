import Foundation

/// A JSON value.
///
/// Use `JSONValue` to pass arbitrary JSON to a provider,
/// such as extra request body fields or custom generation options.
/// It encodes and decodes as plain JSON.
///
/// You can create a JSON value with a literal:
///
/// ```swift
/// let value: JSONValue = [
///     "name": "Ada",
///     "age": 36,
///     "active": true,
///     "tags": ["math", "engines"],
///     "manager": nil,
/// ]
/// ```
public enum JSONValue: Sendable, Hashable {
    /// A JSON null value.
    case null

    /// A JSON boolean value.
    case bool(Bool)

    /// A JSON number without a fractional part.
    case int(Int)

    /// A JSON number with a fractional part.
    case double(Double)

    /// A JSON string value.
    case string(String)

    /// A JSON array.
    case array([JSONValue])

    /// A JSON object.
    case object([String: JSONValue])

    /// Creates a JSON value from an encodable value.
    ///
    /// This initializer encodes the value as JSON and decodes the result.
    /// If the value is already a `JSONValue`, it's returned unchanged.
    ///
    /// - Parameter value: The value to convert.
    /// - Throws: An error if the value can't be encoded as JSON.
    public init<T: Encodable>(_ value: T) throws {
        if let value = value as? JSONValue {
            self = value
        } else {
            let data = try JSONEncoder().encode(value)
            self = try JSONDecoder().decode(JSONValue.self, from: data)
        }
    }

    /// A Boolean value that indicates whether this is a null value.
    public var isNull: Bool {
        self == .null
    }

    /// The value of a boolean value, or `nil` for any other value.
    public var boolValue: Bool? {
        guard case .bool(let value) = self else { return nil }
        return value
    }

    /// The value of an integer value, or `nil` for any other value.
    ///
    /// This property returns `nil` for a ``double(_:)`` value,
    /// even one without a fractional part.
    public var intValue: Int? {
        guard case .int(let value) = self else { return nil }
        return value
    }

    /// The value of a number, or `nil` for any other value.
    ///
    /// This property converts an ``int(_:)`` value to `Double`.
    public var doubleValue: Double? {
        switch self {
        case .double(let value): return value
        case .int(let value): return Double(value)
        default: return nil
        }
    }

    /// The value of a string value, or `nil` for any other value.
    public var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    /// The elements of an array value, or `nil` for any other value.
    public var arrayValue: [JSONValue]? {
        guard case .array(let value) = self else { return nil }
        return value
    }

    /// The dictionary of an object value, or `nil` for any other value.
    public var objectValue: [String: JSONValue]? {
        guard case .object(let value) = self else { return nil }
        return value
    }
}

// MARK: - Codable

extension JSONValue: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Expected a JSON value."
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        }
    }
}

// MARK: - Literals

extension JSONValue: ExpressibleByNilLiteral {
    /// Creates a null JSON value.
    public init(nilLiteral: ()) {
        self = .null
    }
}

extension JSONValue: ExpressibleByBooleanLiteral {
    /// Creates a boolean JSON value.
    public init(booleanLiteral value: Bool) {
        self = .bool(value)
    }
}

extension JSONValue: ExpressibleByIntegerLiteral {
    /// Creates an integer JSON value.
    public init(integerLiteral value: Int) {
        self = .int(value)
    }
}

extension JSONValue: ExpressibleByFloatLiteral {
    /// Creates a floating-point JSON value.
    public init(floatLiteral value: Double) {
        self = .double(value)
    }
}

extension JSONValue: ExpressibleByStringLiteral {
    /// Creates a string JSON value.
    public init(stringLiteral value: String) {
        self = .string(value)
    }
}

extension JSONValue: ExpressibleByArrayLiteral {
    /// Creates an array JSON value.
    public init(arrayLiteral elements: JSONValue...) {
        self = .array(elements)
    }
}

extension JSONValue: ExpressibleByDictionaryLiteral {
    /// Creates an object JSON value.
    ///
    /// If a key appears more than once, the last value wins.
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        var object: [String: JSONValue] = [:]
        for (key, value) in elements {
            object[key] = value
        }
        self = .object(object)
    }
}
