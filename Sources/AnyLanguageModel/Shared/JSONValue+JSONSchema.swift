import enum JSONSchema.JSONValue

// This file imports only `JSONSchema.JSONValue`,
// so `JSONSchema` below names the module, not the `JSONSchema` type.

extension AnyLanguageModel.JSONValue {
    /// Creates a JSON value from the JSONSchema package's JSON value.
    init(_ value: JSONSchema.JSONValue) {
        switch value {
        case .null:
            self = .null
        case .bool(let value):
            self = .bool(value)
        case .int(let value):
            self = .int(value)
        case .double(let value):
            self = .double(value)
        case .string(let value):
            self = .string(value)
        case .array(let values):
            self = .array(values.map { AnyLanguageModel.JSONValue($0) })
        case .object(let object):
            self = .object(object.mapValues { AnyLanguageModel.JSONValue($0) })
        }
    }

    /// This value as the JSONSchema package's JSON value.
    var jsonSchemaValue: JSONSchema.JSONValue {
        switch self {
        case .null:
            return .null
        case .bool(let value):
            return .bool(value)
        case .int(let value):
            return .int(value)
        case .double(let value):
            return .double(value)
        case .string(let value):
            return .string(value)
        case .array(let values):
            return .array(values.map(\.jsonSchemaValue))
        case .object(let object):
            return .object(object.mapValues(\.jsonSchemaValue))
        }
    }
}
