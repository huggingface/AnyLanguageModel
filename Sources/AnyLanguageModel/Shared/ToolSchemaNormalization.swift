import Foundation

/// Inlines JSON Schema `$ref` references against their `$defs`/`definitions` tables.
///
/// `GenerationSchema` encodes a nested schema as a bare `{"$ref": "#/$defs/<name>"}`,
/// and `withResolvedRoot()` resolves only the top-level root reference. Chat
/// templates that do not dereference `$ref` need the nested shape inlined first.
/// Keys declared beside a `$ref` are merged onto the resolved target, and inlined
/// definition tables are dropped. Expansion is bounded: a reference cycle, a pointer
/// this resolver cannot follow, or a schema exceeding the depth or node budget
/// collapses to `{"type": "string"}`. No `$ref` outlives the tables it was resolved
/// against: every subschema slot is walked, and a reference anywhere else collapses
/// too. A resolved `anyOf`/`oneOf` still carries no scalar `"type"`, so follow this
/// with ``normalizeToolSchemaTypes(_:)``.
func resolveToolSchemaRefs(_ schema: [String: Any]) -> [String: Any] {
    resolveSchemaReferences(schema, defs: [:], activePath: [], depth: 0, budget: ExpansionBudget())
}

private let definitionSections = ["$defs", "definitions"]

/// Slots holding one subschema, or — in draft-07 tuple form — an array of them. The
/// tool-spec wrappers `function` and `parameters` nest a schema the same way.
private let subschemaSlots: Set<String> = [
    "additionalItems", "additionalProperties", "allOf", "anyOf", "contains", "else",
    "function", "if", "items", "not", "oneOf", "parameters", "prefixItems",
    "propertyNames", "then", "unevaluatedItems", "unevaluatedProperties",
]

/// Slots mapping names to subschemas.
private let subschemaMapSlots: Set<String> = [
    "dependencies", "dependentSchemas", "patternProperties", "properties",
]

/// Slots whose values are instance data rather than subschemas.
private let instanceDataSlots: Set<String> = ["const", "default", "enum", "examples"]

/// Caps recursion well above any practical schema. Deep nesting overflows the stack,
/// which is not a catchable error, and the cap also bounds the emitted schema's depth
/// for the recursive encoders and templates downstream.
private let maxSchemaDepth = 64

/// Caps how many nodes one schema may expand into. A definition referenced twice by a
/// sibling is a DAG, not a cycle, so the cycle guard alone lets an N-level diamond
/// inline 2^N copies.
private let maxExpandedNodes = 5_000

private var safeScalarSchema: [String: Any] { ["type": "string"] }

private final class ExpansionBudget {
    private var remaining = maxExpandedNodes

    func claimNode() -> Bool {
        guard remaining > 0 else { return false }
        remaining -= 1
        return true
    }
}

private func definitionKey(for ref: String) -> String? {
    guard ref.hasPrefix("#/") else { return nil }
    let pointer = ref.dropFirst(2)
    let components = pointer.split(separator: "/", omittingEmptySubsequences: false)
    guard components.count == 2, definitionSections.contains(String(components[0])) else { return nil }
    return String(pointer)
}

private func resolveSchemaReferences(
    _ node: [String: Any],
    defs inheritedDefs: [String: Any],
    activePath: Set<String>,
    depth: Int,
    budget: ExpansionBudget
) -> [String: Any] {
    guard depth < maxSchemaDepth, budget.claimNode() else { return safeScalarSchema }

    var defs = inheritedDefs
    for section in definitionSections {
        guard let table = node[section] as? [String: Any] else { continue }
        for (name, definition) in table { defs["\(section)/\(name)"] = definition }
    }

    if let ref = node["$ref"] as? String {
        var target = safeScalarSchema
        if let key = definitionKey(for: ref), !activePath.contains(key),
            let definition = defs[key] as? [String: Any]
        {
            target = resolveSchemaReferences(
                definition,
                defs: defs,
                activePath: activePath.union([key]),
                depth: depth + 1,
                budget: budget
            )
        }

        var siblings = node
        for key in ["$ref"] + definitionSections { siblings.removeValue(forKey: key) }
        guard !siblings.isEmpty else { return target }

        let resolvedSiblings = resolveSchemaReferences(
            siblings,
            defs: defs,
            activePath: activePath,
            depth: depth + 1,
            budget: budget
        )
        return target.merging(resolvedSiblings) { _, sibling in sibling }
    }

    var resolved: [String: Any] = [:]
    for (key, value) in node where !definitionSections.contains(key) {
        resolved[key] = resolveSlot(
            key,
            value,
            defs: defs,
            activePath: activePath,
            depth: depth + 1,
            budget: budget
        )
    }
    return resolved
}

/// Resolves `value` as whatever kind of subschema `key` names, and strips references
/// out of a slot that holds no subschema at all. Slots carrying instance data rather
/// than subschemas pass through untouched: a `$ref` key there is a literal value, not
/// a reference, and no template dereferences it.
private func resolveSlot(
    _ key: String,
    _ value: Any,
    defs: [String: Any],
    activePath: Set<String>,
    depth: Int,
    budget: ExpansionBudget
) -> Any {
    if instanceDataSlots.contains(key) {
        return value
    }
    if subschemaSlots.contains(key) {
        return resolveSubschema(value, defs: defs, activePath: activePath, depth: depth, budget: budget)
    }
    if subschemaMapSlots.contains(key), let table = value as? [String: Any] {
        return table.mapValues {
            resolveSubschema($0, defs: defs, activePath: activePath, depth: depth, budget: budget)
        }
    }
    return strippingResidualRefs(value, depth: depth)
}

/// Resolves a subschema, or every element of a tuple-form array of subschemas. A slot
/// carrying a `Bool` — which `additionalProperties` and `items` may — passes through.
private func resolveSubschema(
    _ value: Any,
    defs: [String: Any],
    activePath: Set<String>,
    depth: Int,
    budget: ExpansionBudget
) -> Any {
    if let subschema = value as? [String: Any] {
        return resolveSchemaReferences(subschema, defs: defs, activePath: activePath, depth: depth, budget: budget)
    }
    guard let members = value as? [Any] else { return value }
    guard depth < maxSchemaDepth else { return safeScalarSchema }
    return members.map {
        resolveSubschema($0, defs: defs, activePath: activePath, depth: depth + 1, budget: budget)
    }
}

/// Collapses any mapping still carrying a string `$ref` to ``safeScalarSchema``.
///
/// Definition tables are dropped either way, so a reference in a slot the lists above
/// do not name would reach the template dangling — worse than never resolving it.
private func strippingResidualRefs(_ value: Any, depth: Int) -> Any {
    if let mapping = value as? [String: Any] {
        guard !(mapping["$ref"] is String), depth < maxSchemaDepth else { return safeScalarSchema }
        return mapping.mapValues { strippingResidualRefs($0, depth: depth + 1) }
    }
    if let members = value as? [Any] {
        guard depth < maxSchemaDepth else { return safeScalarSchema }
        return members.map { strippingResidualRefs($0, depth: depth + 1) }
    }
    return value
}

/// Coerces `"type"` values so a chat template can render every property.
///
/// A non-string `"type"` becomes a scalar string — an array-valued one becomes its
/// first non-`"null"` element — and a property or item schema with no `"type"` gets
/// `"string"`. A `"properties"` or `"items"` value that is not a schema mapping is
/// replaced with one, since templates hand those straight to filters that require a
/// mapping. All of these are valid JSON Schema but throw in templates that inspect
/// `"type"` directly. Well-formed schemas pass through unchanged.
func normalizeToolSchemaTypes(_ schema: [String: Any]) -> [String: Any] {
    normalizeSchemaTypes(schema, depth: 0)
}

private func normalizeSchemaTypes(_ schema: [String: Any], depth: Int) -> [String: Any] {
    guard depth < maxSchemaDepth else { return safeScalarSchema }
    var normalized = schema

    if let type = normalized["type"], !(type is String) {
        let named = (type as? [Any])?.compactMap { $0 as? String }.filter { $0 != "null" }
        normalized["type"] = named?.first ?? "string"
    }

    if let properties = normalized["properties"] {
        normalized["properties"] = (properties as? [String: Any] ?? [:]).mapValues { value -> Any in
            guard let schema = value as? [String: Any] else { return safeScalarSchema }
            var property = normalizeSchemaTypes(schema, depth: depth + 1)
            if property["type"] == nil { property["type"] = "string" }
            return property
        }
    }

    if let itemsSchema = normalized["items"] {
        var items =
            (itemsSchema as? [String: Any]).map { normalizeSchemaTypes($0, depth: depth + 1) } ?? safeScalarSchema
        if items["type"] == nil { items["type"] = "string" }
        normalized["items"] = items
    }

    for key in ["function", "parameters"] {
        if let nested = normalized[key] as? [String: Any] {
            normalized[key] = normalizeSchemaTypes(nested, depth: depth + 1)
        }
    }

    return normalized
}
