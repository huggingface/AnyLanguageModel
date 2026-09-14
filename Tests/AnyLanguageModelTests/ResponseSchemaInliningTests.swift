import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Response schema inlining")
struct ResponseSchemaInliningTests {
    @Generable struct Edit { let old: String; let new: String }
    @Generable struct Result { let analysis: String; let edits: [Edit] }
    @Generable struct Inner { let value: String }
    @Generable struct Outer { let note: String; let inner: Inner }

    private func inlined<T: Generable>(_ type: T.Type) throws -> [String: Any] {
        try inlined(T.generationSchema)
    }

    private func inlined(_ schema: GenerationSchema, omitAdditionalProperties: Bool? = nil) throws -> [String: Any] {
        let data = try JSONEncoder().encode(
            try schema.inlinedJSONSchema(omitAdditionalProperties: omitAdditionalProperties)
        )
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func containsRef(_ value: Any) -> Bool {
        if let object = value as? [String: Any] {
            return object["$ref"] != nil || object.values.contains(where: containsRef)
        }
        if let array = value as? [Any] {
            return array.contains(where: containsRef)
        }
        return false
    }

    @Test func inlinesArrayElementDefinitions() throws {
        let schema = try inlined(Result.self)
        #expect(schema["$defs"] == nil)
        #expect(!containsRef(schema))

        let properties = try #require(schema["properties"] as? [String: Any])
        let edits = try #require(properties["edits"] as? [String: Any])
        let items = try #require(edits["items"] as? [String: Any])
        #expect(items["type"] as? String == "object")
        let itemProperties = try #require(items["properties"] as? [String: Any])
        #expect(Set(itemProperties.keys) == ["old", "new"])
    }

    @Test func inlinesNestedObjectDefinitions() throws {
        let schema = try inlined(Outer.self)
        #expect(schema["$defs"] == nil)
        #expect(!containsRef(schema))

        let properties = try #require(schema["properties"] as? [String: Any])
        let inner = try #require(properties["inner"] as? [String: Any])
        #expect(inner["type"] as? String == "object")
        let innerProperties = try #require(inner["properties"] as? [String: Any])
        #expect(innerProperties["value"] != nil)
    }

    @Test func leavesFlatSchemasUnchanged() throws {
        let schema = try inlined(Edit.self)
        #expect(schema["$defs"] == nil)
        #expect(schema["type"] as? String == "object")
        #expect(Set(try #require(schema["required"] as? [String])) == ["old", "new"])
    }

    @Test func preservesFiveThousandIntegerProperties() throws {
        let root = DynamicGenerationSchema(
            name: "LargeResponse",
            properties: (0 ..< 5_000).map {
                .init(name: "p\($0)", schema: .init(type: Int.self))
            }
        )
        let schema = try inlined(GenerationSchema(root: root, dependencies: []))
        let properties = try #require(schema["properties"] as? [String: [String: Any]])

        #expect(properties.count == 5_000)
        #expect(properties.values.allSatisfy { $0["type"] as? String == "integer" })
        #expect(Set(try #require(schema["required"] as? [String])) == Set(properties.keys))
    }

    @Test func inlinesSharedDefinitionsInsideUnions() throws {
        let item = DynamicGenerationSchema(
            name: "Item",
            properties: [.init(name: "value", schema: .init(type: Int.self))]
        )
        let root = DynamicGenerationSchema(
            name: "Response",
            properties: [
                .init(name: "first", schema: .init(referenceTo: "Item")),
                .init(
                    name: "second",
                    schema: .init(name: "Choice", anyOf: [.init(referenceTo: "Item"), .init(type: String.self)])
                ),
            ]
        )
        let schema = try inlined(GenerationSchema(root: root, dependencies: [item]))
        let properties = try #require(schema["properties"] as? [String: [String: Any]])
        let first = try #require(properties["first"])
        let choices = try #require(properties["second"]?["anyOf"] as? [[String: Any]])

        #expect(!containsRef(schema))
        #expect(first["type"] as? String == "object")
        #expect(choices.count == 2)
        #expect(choices.first as NSDictionary? == first as NSDictionary)
        #expect(choices.last?["type"] as? String == "string")
    }

    @Test(arguments: [nil, false, true] as [Bool?])
    func preservesAdditionalPropertiesOption(_ omit: Bool?) throws {
        let schema = try inlined(Outer.generationSchema, omitAdditionalProperties: omit)
        let properties = try #require(schema["properties"] as? [String: [String: Any]])
        let inner = try #require(properties["inner"])
        for object in [schema, inner] {
            if omit == true {
                #expect(object["additionalProperties"] == nil)
            } else {
                #expect(object["additionalProperties"] as? Bool == false)
            }
        }
    }

    @Test func rejectsRecursiveReferences() throws {
        let root = DynamicGenerationSchema(
            name: "Node",
            properties: [
                .init(name: "children", schema: .init(arrayOf: .init(referenceTo: "Node")))
            ]
        )
        let schema = try GenerationSchema(root: root, dependencies: [])
        #expect(throws: GenerationSchema.InliningError.recursiveReference("Node")) {
            try schema.inlinedJSONSchema()
        }
    }

    @Test(arguments: [
        ##"{"$ref":"#/$defs/Missing"}"##,
        ##"{"type":"object","properties":{"item":{"$ref":"#/$defs/Missing"}}}"##,
    ])
    func rejectsUndefinedReferences(_ json: String) throws {
        let schema = try JSONDecoder().decode(GenerationSchema.self, from: Data(json.utf8))
        #expect(throws: GenerationSchema.InliningError.undefinedReference("Missing")) {
            try schema.inlinedJSONSchema()
        }
    }

    @Test func rejectsExcessiveDepth() throws {
        var root = DynamicGenerationSchema(type: Int.self)
        for _ in 0 ..< 100 {
            root = DynamicGenerationSchema(arrayOf: root)
        }
        let schema = try GenerationSchema(root: root, dependencies: [])
        #expect(throws: GenerationSchema.InliningError.depthLimitExceeded) {
            try schema.inlinedJSONSchema()
        }
    }

    @Test func rejectsExcessiveReferenceExpansion() throws {
        let dependencies = (0 ..< 15).map { level in
            DynamicGenerationSchema(
                name: "Level\(level)",
                properties: ["left", "right"].map { name in
                    .init(
                        name: name,
                        schema: level == 14 ? .init(type: Int.self) : .init(referenceTo: "Level\(level + 1)")
                    )
                }
            )
        }
        let schema = try GenerationSchema(root: .init(referenceTo: "Level0"), dependencies: dependencies)
        #expect(throws: GenerationSchema.InliningError.nodeLimitExceeded) {
            try schema.inlinedJSONSchema()
        }
    }
}
