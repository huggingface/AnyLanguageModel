import enum JSONSchema.JSONValue

/// A type-safe representation of JSON values used by AnyLanguageModel APIs.
///
/// - Note: This API is exclusive to AnyLanguageModel
///   and using it means your code is no longer drop-in compatible
///   with the Foundation Models framework.
public typealias JSONValue = JSONSchema.JSONValue
