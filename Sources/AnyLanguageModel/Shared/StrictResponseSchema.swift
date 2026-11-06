extension GenerationSchema {
    /// Converts the schema to a JSON value for OpenAI's strict structured outputs.
    ///
    /// Open Responses providers follow OpenAI's structured outputs,
    /// so they use this too.
    ///
    /// - Throws: An error if the schema can't be inlined or encoded.
    func strictModeJSONValue() throws -> JSONValue {
        // OpenAI's structured outputs don't support `const`.
        let jsonSchema = try inlinedJSONSchema(omitAdditionalProperties: false, constantsAsEnums: true)
        return try JSONValue(jsonSchema).applyingStrictMode()
    }
}

extension JSONValue {
    /// Returns this schema with OpenAI's strict mode requirements applied
    /// to every object in it.
    ///
    /// Strict mode requires each object to have `additionalProperties: false`
    /// and to list all of its properties in `required`.
    /// An optional property, which the schema leaves out of `required`,
    /// becomes required with a type that also allows `null`.
    func applyingStrictMode() -> JSONValue {
        switch self {
        case .array(let values):
            return .array(values.map { $0.applyingStrictMode() })
        case .object(var schema):
            for key in ["items", "anyOf", "oneOf"] {
                schema[key] = schema[key]?.applyingStrictMode()
            }
            if case .object(var properties)? = schema["properties"] {
                let required = Set(schema["required"]?.arrayValue?.compactMap(\.stringValue) ?? [])
                for (name, property) in properties {
                    let strict = property.applyingStrictMode()
                    properties[name] = required.contains(name) ? strict : strict.allowingNull()
                }
                schema["properties"] = .object(properties)
                schema["required"] = .array(properties.keys.sorted().map { .string($0) })
                schema["additionalProperties"] = .bool(false)
            }
            return .object(schema)
        default:
            return self
        }
    }

    /// Returns this schema with `null` also allowed.
    ///
    /// A schema with a single primitive type and no listed values
    /// gets `null` in its `type`, as in OpenAI's examples.
    /// A union gets a `null` choice,
    /// and any other schema becomes one choice of a union with `null`.
    private func allowingNull() -> JSONValue {
        guard case .object(var schema) = self, !allowsNull else { return self }
        let null: JSONValue = .object(["type": .string("null")])
        if case .string(let type)? = schema["type"],
            ["string", "number", "integer", "boolean"].contains(type),
            schema["enum"] == nil, schema["const"] == nil
        {
            schema["type"] = .array([.string(type), .string("null")])
            return .object(schema)
        }
        if case .array(let choices)? = schema["anyOf"], schema.count == 1 {
            return .object(["anyOf": .array(choices + [null])])
        }
        return .object(["anyOf": .array([self, null])])
    }

    /// Whether this schema already allows `null`.
    private var allowsNull: Bool {
        guard case .object(let schema) = self else { return false }
        switch schema["type"] {
        case .string("null"):
            return true
        case .array(let types):
            return types.contains(.string("null"))
        default:
            break
        }
        for key in ["anyOf", "oneOf"] {
            if case .array(let choices)? = schema[key], choices.contains(where: \.allowsNull) {
                return true
            }
        }
        return false
    }
}
