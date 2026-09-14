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
    /// - Throws: An error if a reference is missing or recursive,
    ///   or if inlining exceeds the depth or node limit.
    func inlinedJSONSchema(omitAdditionalProperties: Bool? = nil) throws -> JSONSchema {
        var remainingNodes = maxInlinedSchemaNodes
        let inlined = try inlineReferences(in: root, remainingNodes: &remainingNodes)
        let encoder = JSONEncoder()
        if let omitAdditionalProperties {
            encoder.userInfo[GenerationSchema.omitAdditionalPropertiesKey] = omitAdditionalProperties
        }
        let data = try encoder.encode(inlined)
        return try JSONDecoder().decode(JSONSchema.self, from: data)
    }

    enum InliningError: Error, Equatable, LocalizedError {
        case undefinedReference(String)
        case recursiveReference(String)
        case depthLimitExceeded
        case nodeLimitExceeded

        var errorDescription: String? {
            switch self {
            case .undefinedReference(let name):
                return "The response schema references an undefined type '\(name)'."
            case .recursiveReference(let name):
                return "The response schema contains a recursive reference to '\(name)'."
            case .depthLimitExceeded:
                return "Response schema inlining exceeds the depth limit of \(maxInlinedSchemaDepth)."
            case .nodeLimitExceeded:
                return "Response schema inlining exceeds the node limit of \(maxInlinedSchemaNodes)."
            }
        }
    }

    private func inlineReferences(
        in node: Node,
        activeReferences: Set<String> = [],
        depth: Int = 0,
        remainingNodes: inout Int
    ) throws -> Node {
        guard depth < maxInlinedSchemaDepth else { throw InliningError.depthLimitExceeded }
        guard remainingNodes > 0 else { throw InliningError.nodeLimitExceeded }
        remainingNodes -= 1

        switch node {
        case .ref(let name):
            guard let definition = defs[name] else { throw InliningError.undefinedReference(name) }
            guard !activeReferences.contains(name) else { throw InliningError.recursiveReference(name) }
            return try inlineReferences(
                in: definition,
                activeReferences: activeReferences.union([name]),
                depth: depth + 1,
                remainingNodes: &remainingNodes
            )
        case .object(var object):
            object.properties = try object.properties.mapValues {
                try inlineReferences(
                    in: $0,
                    activeReferences: activeReferences,
                    depth: depth + 1,
                    remainingNodes: &remainingNodes
                )
            }
            return .object(object)
        case .array(var array):
            array.items = try inlineReferences(
                in: array.items,
                activeReferences: activeReferences,
                depth: depth + 1,
                remainingNodes: &remainingNodes
            )
            return .array(array)
        case .anyOf(let choices):
            return .anyOf(
                try choices.map {
                    try inlineReferences(
                        in: $0,
                        activeReferences: activeReferences,
                        depth: depth + 1,
                        remainingNodes: &remainingNodes
                    )
                }
            )
        case .string, .number, .boolean:
            return node
        }
    }
}

// Bound recursion and repeated expansion of shared definitions.
// The node budget includes references and containers,
// with room for a response containing 5,000 scalar properties.
private let maxInlinedSchemaDepth = 64
private let maxInlinedSchemaNodes = 10_000
