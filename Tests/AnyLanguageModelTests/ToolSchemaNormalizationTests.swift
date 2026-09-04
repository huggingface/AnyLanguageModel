import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Tool Schema Normalization")
struct ToolSchemaNormalizationTests {

    // MARK: - Array-Valued Types

    @Test func coercesArrayValuedTypeToFirstNonNullElement() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": [
                    "type": ["string", "null"],
                    "description": "An optional name",
                ]
            ],
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        let name = properties?["name"] as? [String: Any]
        #expect(name?["type"] as? String == "string")
        #expect(name?["description"] as? String == "An optional name")
    }

    @Test func coercesArrayValuedTypeSkippingLeadingNull() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "count": ["type": ["null", "integer"]]
            ],
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        let count = properties?["count"] as? [String: Any]
        #expect(count?["type"] as? String == "integer")
    }

    @Test func coercesAllNullTypeArrayToString() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "value": ["type": ["null"]]
            ],
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        let value = properties?["value"] as? [String: Any]
        #expect(value?["type"] as? String == "string")
    }

    @Test func coercesEmptyTypeArrayToString() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "value": ["type": [String]()]
            ],
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        let value = properties?["value"] as? [String: Any]
        #expect(value?["type"] as? String == "string")
    }

    // MARK: - Missing Types

    @Test func injectsStringTypeIntoPropertyWithoutType() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "note": ["description": "Free-form text"]
            ],
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        let note = properties?["note"] as? [String: Any]
        #expect(note?["type"] as? String == "string")
        #expect(note?["description"] as? String == "Free-form text")
    }

    @Test func injectsStringTypeIntoItemsWithoutType() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "tags": [
                    "type": "array",
                    "items": ["description": "A tag"],
                ]
            ],
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        let tags = properties?["tags"] as? [String: Any]
        let items = tags?["items"] as? [String: Any]
        #expect(items?["type"] as? String == "string")
    }

    @Test func doesNotInjectTypeIntoRootSchema() {
        let schema: [String: Any] = [
            "properties": [
                "name": ["type": "string"]
            ]
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        #expect(normalized["type"] == nil)
    }

    // MARK: - Nesting

    @Test func normalizesDeeplyNestedSchemas() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "entries": [
                    "type": ["array", "null"],
                    "items": [
                        "type": "object",
                        "properties": [
                            "label": ["type": ["string", "null"]],
                            "detail": ["description": "No type declared"],
                        ],
                    ],
                ]
            ],
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        let entries = properties?["entries"] as? [String: Any]
        #expect(entries?["type"] as? String == "array")

        let items = entries?["items"] as? [String: Any]
        let itemProperties = items?["properties"] as? [String: Any]
        let label = itemProperties?["label"] as? [String: Any]
        let detail = itemProperties?["detail"] as? [String: Any]
        #expect(label?["type"] as? String == "string")
        #expect(detail?["type"] as? String == "string")
    }

    @Test func normalizesSchemaNestedInFunctionSpec() {
        let toolSpec: [String: Any] = [
            "type": "function",
            "function": [
                "name": "lookup",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "query": ["type": ["string", "null"]]
                    ],
                ],
            ],
        ]

        let normalized = normalizeToolSchemaTypes(toolSpec)

        let function = normalized["function"] as? [String: Any]
        let parameters = function?["parameters"] as? [String: Any]
        let properties = parameters?["properties"] as? [String: Any]
        let query = properties?["query"] as? [String: Any]
        #expect(query?["type"] as? String == "string")
        #expect(normalized["type"] as? String == "function")
        #expect(function?["name"] as? String == "lookup")
    }

    // MARK: - Pass-Through

    @Test func leavesWellFormedSchemaUnchanged() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": [
                    "type": "string",
                    "description": "A name",
                ],
                "age": [
                    "type": "integer",
                    "minimum": 0,
                ],
                "tags": [
                    "type": "array",
                    "items": ["type": "string"],
                ],
            ],
            "required": ["name"],
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        #expect(normalized as NSDictionary == schema as NSDictionary)
    }

    @Test func leavesNonDictionaryValuesUntouched() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "mode": [
                    "type": "string",
                    "enum": ["fast", "slow"],
                ]
            ],
            "required": ["mode"],
            "additionalProperties": false,
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        #expect(normalized as NSDictionary == schema as NSDictionary)
    }

    // MARK: - Reference Resolution

    @Test func inlinesNestedObjectRefWithRealType() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "filter": ["$ref": "#/$defs/FilterSpec"]
            ],
            "$defs": [
                "FilterSpec": [
                    "type": "object",
                    "properties": ["q": ["type": "string"]],
                ]
            ],
            "required": ["filter"],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let filter = properties?["filter"] as? [String: Any]
        // The nested object regains its real type and properties rather than
        // being flattened to a string.
        #expect(filter?["type"] as? String == "object")
        #expect(filter?["$ref"] == nil)
        let filterProperties = filter?["properties"] as? [String: Any]
        let q = filterProperties?["q"] as? [String: Any]
        #expect(q?["type"] as? String == "string")
        // The now-inlined $defs table is dropped from the emitted schema.
        #expect(resolved["$defs"] == nil)
    }

    @Test func inlinesDeeplyNestedObjectRefs() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "outer": ["$ref": "#/$defs/Outer"]
            ],
            "$defs": [
                "Outer": [
                    "type": "object",
                    "properties": ["inner": ["$ref": "#/$defs/Inner"]],
                ],
                "Inner": [
                    "type": "object",
                    "properties": ["leaf": ["type": "integer"]],
                ],
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let outer = properties?["outer"] as? [String: Any]
        #expect(outer?["type"] as? String == "object")
        let outerProperties = outer?["properties"] as? [String: Any]
        let inner = outerProperties?["inner"] as? [String: Any]
        #expect(inner?["type"] as? String == "object")
        let innerProperties = inner?["properties"] as? [String: Any]
        let leaf = innerProperties?["leaf"] as? [String: Any]
        #expect(leaf?["type"] as? String == "integer")
    }

    @Test func breaksRecursiveRefCycleSafely() {
        // A self-referential schema: TreeNode.child points back at TreeNode.
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "root": ["$ref": "#/$defs/TreeNode"]
            ],
            "$defs": [
                "TreeNode": [
                    "type": "object",
                    "properties": [
                        "value": ["type": "string"],
                        "child": ["$ref": "#/$defs/TreeNode"],
                    ],
                ]
            ],
        ]

        // The point of the test is that this terminates at all.
        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let root = properties?["root"] as? [String: Any]
        #expect(root?["type"] as? String == "object")
        let rootProperties = root?["properties"] as? [String: Any]
        let value = rootProperties?["value"] as? [String: Any]
        #expect(value?["type"] as? String == "string")
        // The recursive edge collapses to a safe scalar instead of inlining
        // forever.
        let child = rootProperties?["child"] as? [String: Any]
        #expect(child?["type"] as? String == "string")
        #expect(child?["properties"] == nil)
    }

    @Test func replacesDanglingRefWithSafeScalar() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "mystery": ["$ref": "#/$defs/DoesNotExist"]
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let mystery = properties?["mystery"] as? [String: Any]
        #expect(mystery?["type"] as? String == "string")
        #expect(mystery?["$ref"] == nil)
    }

    @Test func resolvesRefsAgainstDefsNestedUnderParameters() {
        // Some callers hand over a whole tool spec whose $defs sits beside the
        // properties under `parameters`, not at the document root.
        let toolSpec: [String: Any] = [
            "type": "function",
            "function": [
                "name": "search",
                "parameters": [
                    "type": "object",
                    "properties": ["filter": ["$ref": "#/$defs/FilterSpec"]],
                    "$defs": [
                        "FilterSpec": [
                            "type": "object",
                            "properties": ["q": ["type": "string"]],
                        ]
                    ],
                ],
            ],
        ]

        let resolved = resolveToolSchemaRefs(toolSpec)

        let function = resolved["function"] as? [String: Any]
        let parameters = function?["parameters"] as? [String: Any]
        let properties = parameters?["properties"] as? [String: Any]
        let filter = properties?["filter"] as? [String: Any]
        #expect(filter?["type"] as? String == "object")
        #expect((filter?["properties"] as? [String: Any])?["q"] != nil)
        #expect(parameters?["$defs"] == nil)
    }

    @Test func resolverLeavesRefFreeSchemaByteIdentical() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string", "description": "A name"],
                "age": ["type": "integer", "minimum": 0],
                "tags": ["type": "array", "items": ["type": "string"]],
            ],
            "required": ["name"],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        #expect(resolved as NSDictionary == schema as NSDictionary)
    }

    // MARK: - Resolve-Then-Normalize Pipeline

    @Test func resolvedAnyOfRefGetsScalarTypeAfterNormalization() {
        let modeDef: [String: Any] = ["anyOf": [["const": "fast"], ["const": "slow"], ["type": "null"]]]
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "mode": ["$ref": "#/$defs/Mode"]
            ],
            "$defs": [
                "Mode": modeDef
            ],
        ]

        let prepared = normalizeToolSchemaTypes(resolveToolSchemaRefs(schema))

        let properties = prepared["properties"] as? [String: Any]
        let mode = properties?["mode"] as? [String: Any]
        // The anyOf ref is inlined, then the union leaf — which carries no
        // scalar "type" — gets a renderable one so the template can format it.
        #expect(mode?["type"] as? String == "string")
        #expect(mode?["anyOf"] != nil)
    }

    @Test func coercesArrayValuedTypeAfterResolution() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "item": ["$ref": "#/$defs/Item"]
            ],
            "$defs": [
                "Item": [
                    "type": "object",
                    "properties": ["name": ["type": ["string", "null"]]],
                ]
            ],
        ]

        let prepared = normalizeToolSchemaTypes(resolveToolSchemaRefs(schema))

        let properties = prepared["properties"] as? [String: Any]
        let item = properties?["item"] as? [String: Any]
        let itemProperties = item?["properties"] as? [String: Any]
        let name = itemProperties?["name"] as? [String: Any]
        #expect(name?["type"] as? String == "string")
    }

    @Test func preparesRealMCPToolShapeForTemplate() {
        // The exact shape MCP tools produce: a nested-object $ref and a named
        // anyOf $ref, both bare typeless stubs, plus a $defs table.
        let filterSpecDef: [String: Any] = [
            "type": "object",
            "properties": ["q": ["type": "string"]],
        ]
        let modeDef: [String: Any] = ["anyOf": [["const": "fast"], ["const": "slow"], ["type": "null"]]]
        let parameters: [String: Any] = [
            "type": "object",
            "properties": [
                "filter": ["$ref": "#/$defs/FilterSpec"],
                "mode": ["$ref": "#/$defs/Mode"],
            ],
            "$defs": [
                "FilterSpec": filterSpecDef,
                "Mode": modeDef,
            ],
            "required": ["filter"],
        ]

        let prepared = normalizeToolSchemaTypes(resolveToolSchemaRefs(parameters))

        let properties = prepared["properties"] as? [String: Any]
        let filter = properties?["filter"] as? [String: Any]
        // filter is a real object, not a string-ified stub.
        #expect(filter?["type"] as? String == "object")
        #expect((filter?["properties"] as? [String: Any])?["q"] != nil)
        // mode is a union that now carries a renderable scalar type.
        let mode = properties?["mode"] as? [String: Any]
        #expect(mode?["type"] as? String == "string")
        // No bare refs or $defs remain anywhere the template will walk.
        #expect(prepared["$defs"] == nil)
    }

    @Test func pipelineLeavesWellFormedSchemaByteIdentical() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string", "description": "A name"],
                "age": ["type": "integer", "minimum": 0],
                "tags": ["type": "array", "items": ["type": "string"]],
            ],
            "required": ["name"],
        ]

        let prepared = normalizeToolSchemaTypes(resolveToolSchemaRefs(schema))

        #expect(prepared as NSDictionary == schema as NSDictionary)
    }

    // MARK: - Bounded Expansion

    @Test func boundsDiamondRefExpansion() {
        // Each definition references the next one twice. That is a DAG, not a
        // cycle, so a path-only cycle guard re-inlines every shared subtree and
        // this 2 KB schema expands to millions of nodes.
        let levels = 22
        var defs: [String: Any] = [
            "L\(levels)": [
                "type": "object",
                "properties": ["leaf": ["type": "string"]],
            ]
        ]
        for level in 0 ..< levels {
            defs["L\(level)"] = [
                "type": "object",
                "properties": [
                    "a": ["$ref": "#/$defs/L\(level + 1)"],
                    "b": ["$ref": "#/$defs/L\(level + 1)"],
                ],
            ]
        }
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["root": ["$ref": "#/$defs/L0"]],
            "$defs": defs,
        ]

        // The budget claims a node per resolver call, so bounding the emitted node
        // count bounds the work too. Unbudgeted, this schema emits ~21 million nodes.
        #expect(schemaShape(normalizeToolSchemaTypes(resolveToolSchemaRefs(schema))).nodes < 25_000)
    }

    @Test func boundsDiamondRefExpansionThroughWidenedSlots() {
        // The same diamond, branching through slots the resolver only started walking
        // once `Dict[str, Model]` and `Tuple[Model, ...]` had to resolve: a slot that
        // recurses outside the budget reopens the blow-up the budget exists to stop.
        let levels = 22
        var defs: [String: Any] = ["L\(levels)": ["type": "object", "properties": ["leaf": ["type": "string"]]]]
        for level in 0 ..< levels {
            defs["L\(level)"] = [
                "type": "object",
                "additionalProperties": ["$ref": "#/$defs/L\(level + 1)"],
                "prefixItems": [["$ref": "#/$defs/L\(level + 1)"]],
            ]
        }
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["root": ["$ref": "#/$defs/L0"]],
            "$defs": defs,
        ]

        #expect(schemaShape(normalizeToolSchemaTypes(resolveToolSchemaRefs(schema))).nodes < 25_000)
    }

    @Test func boundsLinearRefChainDepth() {
        let levels = 1_000
        var defs: [String: Any] = [
            "L\(levels)": ["type": "string"]
        ]
        for level in 0 ..< levels {
            defs["L\(level)"] = [
                "type": "object",
                "properties": ["next": ["$ref": "#/$defs/L\(level + 1)"]],
            ]
        }
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["root": ["$ref": "#/$defs/L0"]],
            "$defs": defs,
        ]

        let resolved = resolveToolSchemaRefs(schema)

        // Unbounded recursion here is a stack overflow, not a catchable error,
        // so the resolved schema has to bottom out at a fixed depth.
        #expect(schemaShape(resolved).depth < 300)
    }

    @Test func boundsDeeplyNestedLiteralSchemaDepth() {
        var schema: [String: Any] = ["type": "object", "properties": ["leaf": ["type": "string"]]]
        for _ in 0 ..< 1_000 {
            schema = ["type": "object", "properties": ["next": schema]]
        }

        let normalized = normalizeToolSchemaTypes(schema)

        #expect(schemaShape(normalized).depth < 300)
    }

    // MARK: - Non-String Types

    @Test func coercesNonStringItemsTypeToString() {
        // A numeric "type" is what Gemma's `item_value | map('upper')` chokes on:
        // coercing only array-valued types and only injecting when absent leaves it.
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "x": [
                    "type": "array",
                    "items": ["type": 5],
                ]
            ],
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        let x = properties?["x"] as? [String: Any]
        let items = x?["items"] as? [String: Any]
        #expect(items?["type"] as? String == "string")
    }

    @Test func coercesNonStringPropertyTypesToString() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "flag": ["type": true],
                "nested": ["type": ["kind": "object"]],
                "nulled": ["type": NSNull()],
            ],
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        #expect((properties?["flag"] as? [String: Any])?["type"] as? String == "string")
        #expect((properties?["nested"] as? [String: Any])?["type"] as? String == "string")
        #expect((properties?["nulled"] as? [String: Any])?["type"] as? String == "string")
    }

    // MARK: - Non-Mapping Subschemas

    @Test func replacesNonMappingPropertiesWithEmptyObject() {
        // Templates hand `properties` straight to dictsort; anything but a
        // mapping raises before the model ever sees the tool.
        let schema: [String: Any] = ["type": "object", "properties": "x"]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        #expect(properties != nil)
        #expect(properties?.isEmpty == true)
    }

    @Test func replacesNonMappingItemsWithSafeScalar() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "tags": ["type": "array", "items": "x"],
                "tuple": ["type": "array", "items": [["type": "string"]]],
            ],
        ]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        let tags = properties?["tags"] as? [String: Any]
        let tuple = properties?["tuple"] as? [String: Any]
        #expect((tags?["items"] as? [String: Any])?["type"] as? String == "string")
        #expect((tuple?["items"] as? [String: Any])?["type"] as? String == "string")
    }

    @Test func replacesNonMappingPropertySchemaWithSafeScalar() {
        let schema: [String: Any] = ["type": "object", "properties": ["x": 5]]

        let normalized = normalizeToolSchemaTypes(schema)

        let properties = normalized["properties"] as? [String: Any]
        let x = properties?["x"] as? [String: Any]
        #expect(x?["type"] as? String == "string")
    }

    @Test func replacesNonMappingPropertiesNestedUnderParameters() {
        let toolSpec: [String: Any] = [
            "type": "function",
            "function": [
                "name": "search",
                "parameters": ["type": "object", "properties": "x"],
            ],
        ]

        let prepared = normalizeToolSchemaTypes(resolveToolSchemaRefs(toolSpec))

        let function = prepared["function"] as? [String: Any]
        let parameters = function?["parameters"] as? [String: Any]
        let properties = parameters?["properties"] as? [String: Any]
        #expect(properties?.isEmpty == true)
    }

    // MARK: - Draft-07 Definitions

    @Test func inlinesDraft07DefinitionsRef() {
        // Pydantic v1 and older MCP servers emit draft-07 `definitions`; treating
        // those refs as dangling silently flattens a real object to a string.
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "address": ["$ref": "#/definitions/Address"]
            ],
            "definitions": [
                "Address": [
                    "type": "object",
                    "properties": ["city": ["type": "string"]],
                ]
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let address = properties?["address"] as? [String: Any]
        #expect(address?["type"] as? String == "object")
        #expect((address?["properties"] as? [String: Any])?["city"] != nil)
        #expect(resolved["definitions"] == nil)
    }

    @Test func resolvesDefsAndDefinitionsIndependently() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "modern": ["$ref": "#/$defs/Shape"],
                "legacy": ["$ref": "#/definitions/Shape"],
            ],
            "$defs": ["Shape": ["type": "object", "properties": ["radius": ["type": "number"]]]],
            "definitions": ["Shape": ["type": "object", "properties": ["side": ["type": "integer"]]]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let modern = (properties?["modern"] as? [String: Any])?["properties"] as? [String: Any]
        let legacy = (properties?["legacy"] as? [String: Any])?["properties"] as? [String: Any]
        #expect(modern?["radius"] != nil)
        #expect(legacy?["side"] != nil)
    }

    @Test func replacesUnresolvablePointerWithSafeScalar() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "subschema": ["$ref": "#/$defs/Address/properties/city"],
                "remote": ["$ref": "https://example.com/schema.json#/Address"],
                "bare": ["$ref": "Address"],
            ],
            "$defs": [
                "Address": [
                    "type": "object",
                    "properties": ["city": ["type": "string"]],
                ]
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        for name in ["subschema", "remote", "bare"] {
            let property = properties?[name] as? [String: Any]
            #expect(property?["type"] as? String == "string")
            #expect(property?["$ref"] == nil)
        }
    }

    // MARK: - Reference Siblings

    @Test func mergesSiblingKeysOntoResolvedRef() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "filter": ["$ref": "#/$defs/FilterSpec", "description": "keepme"]
            ],
            "$defs": [
                "FilterSpec": [
                    "type": "object",
                    "properties": ["q": ["type": "string"]],
                ]
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let filter = properties?["filter"] as? [String: Any]
        #expect(filter?["description"] as? String == "keepme")
        #expect(filter?["type"] as? String == "object")
        #expect((filter?["properties"] as? [String: Any])?["q"] != nil)
    }

    @Test func mergesSiblingKeysOntoDanglingRefFallback() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "mystery": ["$ref": "#/$defs/DoesNotExist", "description": "keepme"]
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let mystery = properties?["mystery"] as? [String: Any]
        #expect(mystery?["type"] as? String == "string")
        #expect(mystery?["description"] as? String == "keepme")
    }

    // MARK: - Subschema Slots

    /// A definition table is dropped whether or not its refs were walked, so every slot
    /// that can hold a subschema has to be inlined rather than left behind dangling.
    private func refSchema(slot: String, value: Any) -> [String: Any] {
        [
            "type": "object",
            "properties": ["field": ["type": "object", slot: value]],
            "$defs": ["A": ["type": "object", "properties": ["city": ["type": "string"]]]],
        ]
    }

    private func resolvedSlot(_ slot: String, value: Any) -> Any? {
        let resolved = resolveToolSchemaRefs(refSchema(slot: slot, value: value))
        let properties = resolved["properties"] as? [String: Any]
        return (properties?["field"] as? [String: Any])?[slot]
    }

    /// The `Dict[str, Model]` shape.
    @Test func resolvesRefUnderAdditionalProperties() {
        let value = resolvedSlot("additionalProperties", value: ["$ref": "#/$defs/A"]) as? [String: Any]

        #expect(value?["type"] as? String == "object")
        #expect((value?["properties"] as? [String: Any])?["city"] != nil)
    }

    /// `additionalProperties`, `items` and the `unevaluated*` pair are all allowed to be
    /// a boolean, which is not a mapping and must survive untouched.
    @Test func keepsBooleanSubschemaSlots() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["field": ["type": "string"]],
            "additionalProperties": false,
            "unevaluatedProperties": true,
            "items": false,
        ]

        #expect(resolveToolSchemaRefs(schema) as NSDictionary == schema as NSDictionary)
    }

    /// The 2020-12 `Tuple[Model, int]` shape.
    @Test func resolvesRefsUnderPrefixItems() {
        let members = resolvedSlot("prefixItems", value: [["$ref": "#/$defs/A"], ["type": "integer"]]) as? [Any]

        #expect((members?.first as? [String: Any])?["type"] as? String == "object")
        #expect((members?.last as? [String: Any])?["type"] as? String == "integer")
    }

    /// The draft-07 tuple shape: `items` as an array of subschemas rather than one.
    @Test func resolvesRefsUnderArrayFormItems() {
        let members = resolvedSlot("items", value: [["$ref": "#/$defs/A"], ["type": "integer"]]) as? [Any]

        #expect((members?.first as? [String: Any])?["type"] as? String == "object")
        #expect((members?.last as? [String: Any])?["type"] as? String == "integer")
    }

    @Test func resolvesRefsUnderPatternProperties() {
        let table = resolvedSlot("patternProperties", value: ["^a": ["$ref": "#/$defs/A"]]) as? [String: Any]
        let matched = table?["^a"] as? [String: Any]

        #expect(matched?["type"] as? String == "object")
        #expect((matched?["properties"] as? [String: Any])?["city"] != nil)
    }

    @Test func resolvesRefsUnderDependentSchemas() {
        let table = resolvedSlot("dependentSchemas", value: ["card": ["$ref": "#/$defs/A"]]) as? [String: Any]

        #expect((table?["card"] as? [String: Any])?["type"] as? String == "object")
    }

    @Test func resolvesRefUnderNot() {
        let negated = resolvedSlot("not", value: ["$ref": "#/$defs/A"]) as? [String: Any]

        #expect(negated?["type"] as? String == "object")
        #expect((negated?["properties"] as? [String: Any])?["city"] != nil)
    }

    @Test func resolvesRefsUnderConditionalSlots() {
        for slot in ["if", "then", "else"] {
            let branch = resolvedSlot(slot, value: ["$ref": "#/$defs/A"]) as? [String: Any]
            #expect(branch?["type"] as? String == "object")
        }
    }

    @Test func resolvesRefsUnderRemainingApplicatorSlots() {
        for slot in ["contains", "propertyNames", "unevaluatedItems", "unevaluatedProperties", "additionalItems"] {
            let applied = resolvedSlot(slot, value: ["$ref": "#/$defs/A"]) as? [String: Any]
            #expect(applied?["type"] as? String == "object")
        }
    }

    // MARK: - Residual Reference Sweep

    /// The slot lists cannot name a keyword that does not exist yet, and dropping the
    /// definition table while leaving the reference is worse than not resolving at all.
    @Test func replacesRefInUnknownSlotWithSafeScalar() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["field": ["type": "string"]],
            "x-future-2030-keyword": ["nested": ["$ref": "#/$defs/A"]],
            "$defs": ["A": ["type": "object", "properties": ["city": ["type": "string"]]]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let future = (resolved["x-future-2030-keyword"] as? [String: Any])?["nested"] as? [String: Any]
        #expect(future?["type"] as? String == "string")
        #expect(!containsRef(resolved))
    }

    /// A known slot holding the wrong shape falls through to the sweep too, so no walk
    /// this resolver skips can hand the template a reference.
    @Test func replacesRefUnderMalformedSubschemaSlot() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [["$ref": "#/$defs/A"]],
            "$defs": ["A": ["type": "object", "properties": ["city": ["type": "string"]]]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let entry = (resolved["properties"] as? [Any])?.first as? [String: Any]
        #expect(entry?["type"] as? String == "string")
        #expect(!containsRef(resolved))
    }

    /// The sweep replaces a reference, not every mapping it walks past.
    @Test func sweepLeavesRefFreeUnknownSlotsUntouched() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["field": ["type": "string"]],
            "x-vendor": ["nested": ["keep": 1], "flag": true, "list": [1, 2]],
            "default": NSNull(),
        ]

        #expect(resolveToolSchemaRefs(schema) as NSDictionary == schema as NSDictionary)
    }

    @Test func keepsInstanceDataContainingRefKeyVerbatim() {
        // const/default/enum/examples carry instance values, not subschemas, so a
        // "$ref" key inside them is literal data the template renders as-is.
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "document": [
                    "type": "object",
                    "default": ["$ref": "#/$defs/Absent"],
                    "examples": [["$ref": "https://example.com/a"]],
                ],
                "pointer": [
                    "type": "string",
                    "const": ["$ref": "#/$defs/Absent"],
                    "enum": [["$ref": "#/$defs/Absent"], "plain"],
                ],
            ],
            "$defs": ["Used": ["type": "string"]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let document = properties?["document"] as? [String: Any]
        let pointer = properties?["pointer"] as? [String: Any]
        #expect(document?["default"] as? NSDictionary == ["$ref": "#/$defs/Absent"] as NSDictionary)
        #expect(document?["examples"] as? NSArray == [["$ref": "https://example.com/a"]] as NSArray)
        #expect(pointer?["const"] as? NSDictionary == ["$ref": "#/$defs/Absent"] as NSDictionary)
        #expect((pointer?["enum"] as? [Any])?.count == 2)
        #expect(resolved["$defs"] == nil)
    }

    // MARK: - Conformance (prior art)
    //
    // Cases ported from established `$ref`/`$defs` resolvers. Each asserts OUR
    // documented contract — inline, siblings win, drop the tables, degrade to
    // `{"type": "string"}` when a pointer cannot be followed — which is not always
    // the source's contract. Every deliberate divergence is called out inline.

    // MARK: Conformance - jsonref (gazpachoking/jsonref)

    /// jsonref `test_non_string_is_not_ref`: a non-string `$ref` is not a reference.
    @Test func jsonrefNonStringRefIsNotAReference() {
        let schema: [String: Any] = ["$ref": [1]]

        let resolved = resolveToolSchemaRefs(schema)

        #expect((resolved["$ref"] as? [Any])?.count == 1)
    }

    /// jsonref `test_merge_extra_flag` / `test_extra_ref_attributes` under
    /// `merge_props=True`, which is the mode our contract matches. jsonref's DEFAULT
    /// (`merge_props=False`) discards the siblings instead; we deliberately do not.
    @Test func jsonrefMergesSiblingPropsOntoTarget() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["b": ["$ref": "#/$defs/a", "extra": 2]],
            "$defs": ["a": ["main": 1]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let b = (resolved["properties"] as? [String: Any])?["b"] as? [String: Any]
        #expect(b?["main"] as? Int == 1)
        #expect(b?["extra"] as? Int == 2)
        #expect(b?["$ref"] == nil)
        #expect(resolved["$defs"] == nil)
    }

    /// jsonref `test_refs_inside_extra_props`: jsonref resolves a `$ref` wherever it
    /// appears in the document. Divergence/scope: we inline only within JSON Schema
    /// subschema slots, so a ref parked under an arbitrary key degrades instead. It
    /// cannot be left as it is, because the definition table is dropped regardless.
    @Test func jsonrefRefUnderArbitraryKeyDegradesInsteadOfResolving() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["q": ["type": "string"]],
            "x-vendor": ["$ref": "#/$defs/a"],
            "$defs": ["a": ["type": "object", "properties": ["main": ["type": "integer"]]]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let vendor = resolved["x-vendor"] as? [String: Any]
        #expect(vendor?["type"] as? String == "string")
        #expect(vendor?["$ref"] == nil)
        #expect(resolved["$defs"] == nil)
    }

    /// jsonref `test_merge_extra_flag` key collision: `{**target, **siblings}`, so a
    /// sibling shadows the same key on the target.
    @Test func jsonrefSiblingWinsOverDefinitionKey() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "x": ["$ref": "#/$defs/D", "description": "from-sibling"]
            ],
            "$defs": [
                "D": [
                    "type": "object",
                    "description": "from-def",
                    "properties": ["q": ["type": "string"]],
                ]
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let x = properties?["x"] as? [String: Any]
        #expect(x?["description"] as? String == "from-sibling")
        #expect(x?["type"] as? String == "object")
    }

    /// jsonref `test_separate_extras`: a ref whose target is itself a ref-with-siblings
    /// accumulates both sibling sets.
    @Test func jsonrefChainedRefsAccumulateSiblings() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "x": ["$ref": "#/$defs/a", "extrax": "x"],
                "z": ["$ref": "#/$defs/y", "extraz": "z"],
            ],
            "$defs": [
                "a": ["main": 1234],
                "y": ["$ref": "#/$defs/a", "extray": "y"],
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let x = properties?["x"] as? [String: Any]
        let z = properties?["z"] as? [String: Any]
        #expect(x?["main"] as? Int == 1234)
        #expect(x?["extrax"] as? String == "x")
        #expect(x?["extray"] == nil)
        #expect(z?["main"] as? Int == 1234)
        #expect(z?["extray"] as? String == "y")
        #expect(z?["extraz"] as? String == "z")
    }

    /// jsonref `test_self_referent_reference_w_merge`: a document that is itself a ref
    /// with siblings. Divergence: jsonref keeps the definition table as an ordinary key,
    /// we drop it by contract because templates would walk it.
    @Test func jsonrefRootLevelRefMergesSiblingsAndDropsTable() {
        let schema: [String: Any] = [
            "$ref": "#/$defs/sub",
            "extra": "aoeu",
            "$defs": ["sub": ["main": "aoeu"]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        #expect(resolved["main"] as? String == "aoeu")
        #expect(resolved["extra"] as? String == "aoeu")
        #expect(resolved["$defs"] == nil)
        #expect(resolved["$ref"] == nil)
    }

    /// jsonref `test_repr_expands_deep_refs_by_default`: a chain of refs collapses to
    /// the value at the end of the chain.
    @Test func jsonrefDeepRefChainFullyExpands() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["f": ["$ref": "#/$defs/f"]],
            "$defs": [
                "a": ["type": "string"],
                "b": ["$ref": "#/$defs/a"],
                "c": ["$ref": "#/$defs/b"],
                "d": ["$ref": "#/$defs/c"],
                "e": ["$ref": "#/$defs/d"],
                "f": ["$ref": "#/$defs/e"],
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let f = properties?["f"] as? [String: Any]
        #expect(f?["type"] as? String == "string")
        #expect(f?["$ref"] == nil)
    }

    /// jsonref `test_no_lazy_load_recursive`: mutual recursion between two definitions
    /// must terminate. Divergence: jsonref builds a cyclic object graph, we cannot —
    /// the back-edge collapses to a safe scalar.
    @Test func jsonrefMutuallyRecursiveDefinitionsTerminate() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["root": ["$ref": "#/$defs/A"]],
            "$defs": [
                "A": ["type": "object", "properties": ["b": ["$ref": "#/$defs/B"]]],
                "B": ["type": "object", "properties": ["a": ["$ref": "#/$defs/A"]]],
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let root = properties?["root"] as? [String: Any]
        let b = (root?["properties"] as? [String: Any])?["b"] as? [String: Any]
        let a = (b?["properties"] as? [String: Any])?["a"] as? [String: Any]
        #expect(root?["type"] as? String == "object")
        #expect(b?["type"] as? String == "object")
        #expect(a?["type"] as? String == "string")
    }

    /// jsonref `test_actual_references_not_copies`: jsonref aliases one object at every
    /// use site. Divergence: we inline an independent copy per site, which is the whole
    /// point — a template cannot follow an alias. Both sites must carry full content.
    @Test func jsonrefSharedTargetIsInlinedAtEverySite() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "b": ["$ref": "#/$defs/a"],
                "c": ["$ref": "#/$defs/a"],
            ],
            "$defs": ["a": ["type": "object", "properties": ["q": ["type": "string"]]]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        for name in ["b", "c"] {
            let site = properties?[name] as? [String: Any]
            #expect(site?["type"] as? String == "object")
            #expect((site?["properties"] as? [String: Any])?["q"] != nil)
        }
    }

    /// jsonref `test_local_nonexistent_ref`. Divergence: jsonref raises `JsonRefError`,
    /// we degrade — a tool that renders is worth more than a turn that throws.
    @Test func jsonrefNonexistentRefsDegradeInsteadOfRaising() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "a": ["$ref": "#/x"],
                "b": ["$ref": "#/0"],
                "c": ["$ref": "#/data/3"],
                "d": ["$ref": "#/$defs/nope"],
            ],
            "$defs": ["data": ["type": "string"]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        for name in ["a", "b", "c", "d"] {
            #expect((properties?[name] as? [String: Any])?["type"] as? String == "string")
        }
    }

    /// jsonref `test_recursive_extra`: `{"$ref": "#"}` points at the whole document.
    /// Divergence/scope: we resolve only `#/$defs/<name>` and `#/definitions/<name>`,
    /// so a whole-document pointer degrades. Siblings still merge onto the fallback.
    @Test func jsonrefWholeDocumentPointerIsOutOfScope() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["a": ["$ref": "#", "extra": "foo"]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let a = properties?["a"] as? [String: Any]
        #expect(a?["type"] as? String == "string")
        #expect(a?["extra"] as? String == "foo")
    }

    /// jsonref `test_local_escaped_ref`: `~1` escapes `/` and `~0` escapes `~` in a JSON
    /// Pointer. KNOWN GAP: we compare pointer segments verbatim, so a definition whose
    /// name contains either character never resolves. No generator we consume emits one.
    @Test func jsonrefEscapedPointerSegmentIsNotUnescaped() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "slash": ["$ref": "#/$defs/a~1a"],
                "tilde": ["$ref": "#/$defs/b~0b"],
            ],
            "$defs": [
                "a/a": ["type": "object", "properties": ["q": ["type": "string"]]],
                "b~b": ["type": "object", "properties": ["q": ["type": "string"]]],
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        #expect((properties?["slash"] as? [String: Any])?["type"] as? String == "string")
        #expect((properties?["tilde"] as? [String: Any])?["type"] as? String == "string")
    }

    // MARK: Conformance - llama.cpp (tests/test-json-schema-to-grammar.cpp)

    /// llama.cpp `top-level $ref`.
    @Test func llamaTopLevelRefIntoDefinitions() {
        let schema: [String: Any] = [
            "$ref": "#/definitions/foo",
            "definitions": [
                "foo": [
                    "type": "object",
                    "properties": ["a": ["type": "string"]],
                    "required": ["a"],
                    "additionalProperties": false,
                ]
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        #expect(resolved["type"] as? String == "object")
        #expect(((resolved["properties"] as? [String: Any])?["a"] as? [String: Any])?["type"] as? String == "string")
        #expect(resolved["definitions"] == nil)
        #expect(resolved["$ref"] == nil)
    }

    /// llama.cpp `$ref to string` / `$ref to integer`: a ref whose target is a scalar.
    @Test func llamaRefToScalarDefinition() {
        let toString = resolveToolSchemaRefs(["$ref": "#/$defs/str", "$defs": ["str": ["type": "string"]]])
        let toInteger = resolveToolSchemaRefs(["$ref": "#/$defs/num", "$defs": ["num": ["type": "integer"]]])

        #expect(toString["type"] as? String == "string")
        #expect(toInteger["type"] as? String == "integer")
        #expect(toString["$defs"] == nil)
        #expect(toInteger["$defs"] == nil)
    }

    /// llama.cpp `anyOf`: a root `anyOf` of definition refs, with a `type` sibling.
    @Test func llamaAnyOfOfDefinitionRefs() {
        let schema: [String: Any] = [
            "anyOf": [
                ["$ref": "#/definitions/foo"],
                ["$ref": "#/definitions/bar"],
            ],
            "definitions": [
                "foo": ["properties": ["a": ["type": "number"]]],
                "bar": ["properties": ["b": ["type": "number"]]],
            ],
            "type": "object",
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let members = resolved["anyOf"] as? [Any]
        let foo = members?.first as? [String: Any]
        let bar = members?.last as? [String: Any]
        #expect((foo?["properties"] as? [String: Any])?["a"] != nil)
        #expect((bar?["properties"] as? [String: Any])?["b"] != nil)
        #expect(resolved["type"] as? String == "object")
        #expect(resolved["definitions"] == nil)
    }

    /// llama.cpp `mix of allOf, anyOf and $ref` (tsconfig.json shape).
    @Test func llamaMixOfAllOfAnyOfAndRef() {
        let schema: [String: Any] = [
            "allOf": [
                ["$ref": "#/definitions/foo"],
                ["$ref": "#/definitions/bar"],
                ["anyOf": [["$ref": "#/definitions/baz"], ["$ref": "#/definitions/bam"]]],
            ],
            "definitions": [
                "foo": ["properties": ["a": ["type": "number"]]],
                "bar": ["properties": ["b": ["type": "number"]]],
                "bam": ["properties": ["c": ["type": "number"]]],
                "baz": ["properties": ["d": ["type": "number"]]],
            ],
            "type": "object",
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let members = resolved["allOf"] as? [Any]
        let foo = members?[0] as? [String: Any]
        let nested = (members?[2] as? [String: Any])?["anyOf"] as? [Any]
        let baz = nested?.first as? [String: Any]
        #expect((foo?["properties"] as? [String: Any])?["a"] != nil)
        #expect((baz?["properties"] as? [String: Any])?["d"] != nil)
        #expect(resolved["definitions"] == nil)
    }

    /// llama.cpp `allOf with enum schema`: the ref target's `enum` survives inlining.
    @Test func llamaAllOfWithEnumSchemaRef() {
        let schema: [String: Any] = [
            "allOf": [["$ref": "#/definitions/foo"]],
            "definitions": ["foo": ["type": "string", "enum": ["a", "b"]]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let foo = (resolved["allOf"] as? [Any])?.first as? [String: Any]
        #expect(foo?["type"] as? String == "string")
        #expect((foo?["enum"] as? [String])?.count == 2)
        #expect(resolved["definitions"] == nil)
    }

    /// llama.cpp emits `"definitions": {}` on several fixtures; an empty table still
    /// has to be stripped or the template renders a stray key.
    @Test func llamaEmptyDefinitionsTableIsDropped() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["a": ["type": "string"]],
            "required": ["a"],
            "additionalProperties": false,
            "definitions": [String: Any](),
        ]

        let resolved = resolveToolSchemaRefs(schema)

        #expect(resolved["definitions"] == nil)
        #expect((resolved["properties"] as? [String: Any])?["a"] != nil)
    }

    /// llama.cpp `anyOf $ref`: a pointer into `properties`, not into a definition table.
    /// Divergence/scope: llama.cpp resolves arbitrary JSON Pointers, we resolve only
    /// definition-table pointers, so this degrades. Here the degraded value happens to
    /// equal the true target (`{"type": "string"}`); a pointer at an object target would
    /// not be so lucky — see `replacesUnresolvablePointerWithSafeScalar`.
    @Test func llamaPointerIntoPropertiesIsOutOfScope() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "a": ["anyOf": [["type": "string"], ["type": "number"]]],
                "b": ["anyOf": [["$ref": "#/properties/a/anyOf/0"], ["type": "boolean"]]],
            ],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let properties = resolved["properties"] as? [String: Any]
        let b = (properties?["b"] as? [String: Any])?["anyOf"] as? [Any]
        #expect((b?.first as? [String: Any])?["type"] as? String == "string")
        #expect((b?.first as? [String: Any])?["$ref"] == nil)
    }

    // MARK: Conformance - haystack (deepset-ai/haystack#10705)

    /// haystack `test_resolve_schema_refs_no_defs`.
    @Test func haystackNoDefsSchemaIsReturnedAsIs() {
        let schema: [String: Any] = ["type": "object", "properties": ["name": ["type": "string"]]]

        #expect(resolveToolSchemaRefs(schema) as NSDictionary == schema as NSDictionary)
    }

    /// haystack `test_resolve_schema_refs_expands_defs`.
    @Test func haystackExpandsDefs() {
        let schema: [String: Any] = [
            "$defs": [
                "User": [
                    "type": "object",
                    "properties": ["name": ["type": "string"], "age": ["type": "integer"]],
                    "required": ["name"],
                ]
            ],
            "type": "object",
            "properties": ["user": ["$ref": "#/$defs/User"]],
            "required": ["user"],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let user = (resolved["properties"] as? [String: Any])?["user"] as? [String: Any]
        #expect(resolved["$defs"] == nil)
        #expect(user?["$ref"] == nil)
        #expect(user?["type"] as? String == "object")
        #expect(((user?["properties"] as? [String: Any])?["name"] as? [String: Any])?["type"] as? String == "string")
        #expect((user?["required"] as? [String])?.first == "name")
    }

    /// haystack `test_resolve_schema_refs_nested_refs`.
    @Test func haystackExpandsNestedRefs() {
        let schema: [String: Any] = [
            "$defs": [
                "Address": ["type": "object", "properties": ["street": ["type": "string"]]],
                "User": [
                    "type": "object",
                    "properties": ["name": ["type": "string"], "address": ["$ref": "#/$defs/Address"]],
                ],
            ],
            "type": "object",
            "properties": ["user": ["$ref": "#/$defs/User"]],
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let user = (resolved["properties"] as? [String: Any])?["user"] as? [String: Any]
        let address = (user?["properties"] as? [String: Any])?["address"] as? [String: Any]
        let street = (address?["properties"] as? [String: Any])?["street"] as? [String: Any]
        #expect(resolved["$defs"] == nil)
        #expect(address?["type"] as? String == "object")
        #expect(street?["type"] as? String == "string")
    }

    /// haystack `test_convert_tools_to_hfapi_tools_resolves_defs`: the same resolution
    /// applied to a tool's `parameters`.
    @Test func haystackResolvesDefsInsideToolParameters() {
        let toolSpec: [String: Any] = [
            "type": "function",
            "function": [
                "name": "get_user",
                "description": "Get user info",
                "parameters": [
                    "$defs": ["User": ["type": "object", "properties": ["name": ["type": "string"]]]],
                    "type": "object",
                    "properties": ["user": ["$ref": "#/$defs/User"]],
                ],
            ],
        ]

        let resolved = resolveToolSchemaRefs(toolSpec)

        let parameters = (resolved["function"] as? [String: Any])?["parameters"] as? [String: Any]
        let user = (parameters?["properties"] as? [String: Any])?["user"] as? [String: Any]
        #expect(parameters?["$defs"] == nil)
        #expect(user?["type"] as? String == "object")
    }

    // MARK: Conformance - Pydantic (real `model_json_schema()` output, v2.13.4)

    /// `class User(BaseModel): name: str; age: int; address: Address`
    @Test func pydanticNestedModelSchema() {
        let schema: [String: Any] = [
            "$defs": [
                "Address": [
                    "properties": [
                        "street": ["title": "Street", "type": "string"],
                        "city": ["title": "City", "type": "string"],
                        "zip_code": [
                            "anyOf": [["type": "string"], ["type": "null"]],
                            "default": NSNull(),
                            "title": "Zip Code",
                        ],
                    ],
                    "required": ["street", "city"],
                    "title": "Address",
                    "type": "object",
                ]
            ],
            "properties": [
                "name": ["title": "Name", "type": "string"],
                "age": ["title": "Age", "type": "integer"],
                "address": ["$ref": "#/$defs/Address"],
            ],
            "required": ["name", "age", "address"],
            "title": "User",
            "type": "object",
        ]

        let prepared = normalizeToolSchemaTypes(resolveToolSchemaRefs(schema))

        let address = (prepared["properties"] as? [String: Any])?["address"] as? [String: Any]
        #expect(address?["type"] as? String == "object")
        #expect(address?["title"] as? String == "Address")
        #expect((address?["properties"] as? [String: Any])?["street"] != nil)
        #expect(prepared["$defs"] == nil)
        #expect(!containsRef(prepared))
    }

    /// `mode: Mode = Mode.fast` emits `{"$ref": ..., "default": "fast"}` — Pydantic's
    /// canonical ref-with-sibling. Dropping the sibling would lose the default.
    @Test func pydanticRefWithDefaultSiblingKeepsBoth() {
        let schema: [String: Any] = [
            "$defs": ["Mode": ["enum": ["fast", "slow"], "title": "Mode", "type": "string"]],
            "properties": ["mode": ["$ref": "#/$defs/Mode", "default": "fast"]],
            "title": "Query",
            "type": "object",
        ]

        let resolved = resolveToolSchemaRefs(schema)

        let mode = (resolved["properties"] as? [String: Any])?["mode"] as? [String: Any]
        #expect(mode?["type"] as? String == "string")
        #expect(mode?["default"] as? String == "fast")
        #expect((mode?["enum"] as? [String])?.count == 2)
        #expect(mode?["$ref"] == nil)
    }

    /// `address: Optional[Address] = None` emits `anyOf: [{$ref}, {"type": "null"}]`.
    @Test func pydanticOptionalNestedModelResolvesInsideAnyOf() {
        let schema: [String: Any] = [
            "$defs": [
                "Address": [
                    "properties": ["city": ["title": "City", "type": "string"]],
                    "required": ["city"],
                    "title": "Address",
                    "type": "object",
                ]
            ],
            "properties": [
                "address": [
                    "anyOf": [["$ref": "#/$defs/Address"], ["type": "null"]],
                    "default": NSNull(),
                ]
            ],
            "title": "Query",
            "type": "object",
        ]

        let prepared = normalizeToolSchemaTypes(resolveToolSchemaRefs(schema))

        let address = (prepared["properties"] as? [String: Any])?["address"] as? [String: Any]
        let members = address?["anyOf"] as? [Any]
        #expect((members?.first as? [String: Any])?["type"] as? String == "object")
        #expect(((members?.first as? [String: Any])?["properties"] as? [String: Any])?["city"] != nil)
        // The union leaf carries no scalar type of its own until normalization adds one.
        #expect(address?["type"] as? String == "string")
        #expect(!containsRef(prepared))
    }

    /// `class Node(BaseModel): value: str; child: Optional["Node"]` — the whole document
    /// is a `$ref` beside its own `$defs`, and the definition is self-recursive.
    @Test func pydanticRecursiveModelSchemaTerminates() {
        let schema: [String: Any] = [
            "$defs": [
                "Node": [
                    "properties": [
                        "value": ["title": "Value", "type": "string"],
                        "child": [
                            "anyOf": [["$ref": "#/$defs/Node"], ["type": "null"]],
                            "default": NSNull(),
                        ],
                    ],
                    "required": ["value"],
                    "title": "Node",
                    "type": "object",
                ]
            ],
            "$ref": "#/$defs/Node",
        ]

        let resolved = resolveToolSchemaRefs(schema)

        #expect(resolved["type"] as? String == "object")
        let child = (resolved["properties"] as? [String: Any])?["child"] as? [String: Any]
        let members = child?["anyOf"] as? [Any]
        #expect((members?.first as? [String: Any])?["type"] as? String == "string")
        #expect((members?.last as? [String: Any])?["type"] as? String == "null")
        #expect(!containsRef(resolved))
    }

    /// `animal: Union[Cat, Dog]` plus `pets: List[Address]` — refs under `anyOf` and
    /// under `items` in one schema.
    @Test func pydanticUnionAndListOfModelResolve() {
        let schema: [String: Any] = [
            "$defs": [
                "Address": ["properties": ["city": ["type": "string"]], "title": "Address", "type": "object"],
                "Cat": ["properties": ["meow": ["type": "string"]], "title": "Cat", "type": "object"],
                "Dog": ["properties": ["woof": ["type": "string"]], "title": "Dog", "type": "object"],
            ],
            "properties": [
                "animal": ["anyOf": [["$ref": "#/$defs/Cat"], ["$ref": "#/$defs/Dog"]], "title": "Animal"],
                "pets": ["items": ["$ref": "#/$defs/Address"], "title": "Pets", "type": "array"],
            ],
            "required": ["animal", "pets"],
            "title": "Pet",
            "type": "object",
        ]

        let prepared = normalizeToolSchemaTypes(resolveToolSchemaRefs(schema))

        let properties = prepared["properties"] as? [String: Any]
        let animal = (properties?["animal"] as? [String: Any])?["anyOf"] as? [Any]
        #expect(((animal?.first as? [String: Any])?["properties"] as? [String: Any])?["meow"] != nil)
        #expect(((animal?.last as? [String: Any])?["properties"] as? [String: Any])?["woof"] != nil)
        let items = (properties?["pets"] as? [String: Any])?["items"] as? [String: Any]
        #expect(items?["type"] as? String == "object")
        #expect((items?["properties"] as? [String: Any])?["city"] != nil)
        #expect(!containsRef(prepared))
    }

    /// `by_id: Dict[str, Address]` emits the `$ref` under `additionalProperties`.
    @Test func pydanticDictOfModelRefResolves() {
        let schema: [String: Any] = [
            "$defs": [
                "Address": [
                    "properties": ["city": ["title": "City", "type": "string"]],
                    "required": ["city"],
                    "title": "Address",
                    "type": "object",
                ]
            ],
            "properties": [
                "by_id": ["additionalProperties": ["$ref": "#/$defs/Address"], "title": "By Id", "type": "object"]
            ],
            "required": ["by_id"],
            "title": "Book",
            "type": "object",
        ]

        let prepared = normalizeToolSchemaTypes(resolveToolSchemaRefs(schema))

        let byID = (prepared["properties"] as? [String: Any])?["by_id"] as? [String: Any]
        let value = byID?["additionalProperties"] as? [String: Any]
        #expect(value?["type"] as? String == "object")
        #expect((value?["properties"] as? [String: Any])?["city"] != nil)
        #expect(!containsRef(prepared))
    }

    /// `pair: Tuple[Address, int]` emits the `$ref` under `prefixItems`.
    @Test func pydanticTupleModelRefResolves() {
        let schema: [String: Any] = [
            "$defs": [
                "Address": [
                    "properties": ["city": ["title": "City", "type": "string"]],
                    "required": ["city"],
                    "title": "Address",
                    "type": "object",
                ]
            ],
            "properties": [
                "pair": [
                    "maxItems": 2,
                    "minItems": 2,
                    "prefixItems": [["$ref": "#/$defs/Address"], ["type": "integer"]],
                    "title": "Pair",
                    "type": "array",
                ]
            ],
            "required": ["pair"],
            "title": "Book",
            "type": "object",
        ]

        let prepared = normalizeToolSchemaTypes(resolveToolSchemaRefs(schema))

        let pair = (prepared["properties"] as? [String: Any])?["pair"] as? [String: Any]
        let members = pair?["prefixItems"] as? [Any]
        #expect((members?.first as? [String: Any])?["type"] as? String == "object")
        #expect(((members?.first as? [String: Any])?["properties"] as? [String: Any])?["city"] != nil)
        #expect((members?.last as? [String: Any])?["type"] as? String == "integer")
        #expect(!containsRef(prepared))
    }

    // MARK: - Helpers

    /// True when any mapping anywhere in the tree still carries a string `$ref`.
    private func containsRef(_ root: Any) -> Bool {
        var pending: [Any] = [root]
        while let value = pending.popLast() {
            if let mapping = value as? [String: Any] {
                if mapping["$ref"] is String { return true }
                pending.append(contentsOf: mapping.values)
            } else if let elements = value as? [Any] {
                pending.append(contentsOf: elements)
            }
        }
        return false
    }

    /// Walks a schema iteratively so a pathological tree cannot overflow the
    /// test's own stack: `nodes` counts mappings, `depth` counts container levels.
    private func schemaShape(_ root: Any) -> (nodes: Int, depth: Int) {
        var pending: [(value: Any, level: Int)] = [(root, 1)]
        var nodes = 0
        var depth = 0
        while let (value, level) = pending.popLast() {
            depth = max(depth, level)
            if let mapping = value as? [String: Any] {
                nodes += 1
                pending.append(contentsOf: mapping.values.map { ($0, level + 1) })
            } else if let elements = value as? [Any] {
                pending.append(contentsOf: elements.map { ($0, level + 1) })
            }
        }
        return (nodes, depth)
    }
}
