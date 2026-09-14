import Foundation
import JSONSchema

extension GenerationSchema {
    /// Converts the schema to a `JSONSchema` with every `$ref` inlined.
    ///
    /// `JSONSchema` has no `$defs` table,
    /// so decoding a schema with nested `@Generable` types
    /// would leave their `$ref`s dangling.
    /// Providers that send the schema over the wire use this instead.
    ///
    /// - Parameter omitAdditionalProperties: Overrides the encoder's
    ///   `additionalProperties` handling when set.
    func inlinedJSONSchema(omitAdditionalProperties: Bool? = nil) throws -> JSONSchema {
        let encoder = JSONEncoder()
        if let omitAdditionalProperties {
            encoder.userInfo[GenerationSchema.omitAdditionalPropertiesKey] = omitAdditionalProperties
        }
        let data = try encoder.encode(withResolvedRoot() ?? self)

        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return try JSONDecoder().decode(JSONSchema.self, from: data)
        }
        let inlined = resolveToolSchemaRefs(object)
        let inlinedData = try JSONSerialization.data(withJSONObject: inlined)
        return try JSONDecoder().decode(JSONSchema.self, from: inlinedData)
    }
}
