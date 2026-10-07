import Foundation
import Testing

@testable import AnyLanguageModel

@Generable
struct StringChoiceGuided {
    @Guide(.constant("fixed"))
    var kind: String

    @Guide(description: "A described constant", .constant("fixed"))
    var described: String

    @Guide(.anyOf(["a", "b"]))
    var choice: String

    @Guide(.anyOf(["x", "y"]))
    var maybe: String?
}

/// `.constant(_:)` and `.anyOf(_:)` string guides constrain the schema,
/// and a constant encodes as `const` unless a provider needs a one-choice `enum`.
@Suite("String choice guides")
struct StringChoiceGuideTests {
    private func encode(
        _ schema: GenerationSchema,
        constantsAsEnums: Bool? = nil
    ) throws -> [String: Any] {
        let encoder = JSONEncoder()
        if let constantsAsEnums {
            encoder.userInfo[GenerationSchema.constantsAsEnumsKey] = constantsAsEnums
        }
        let resolved = schema.withResolvedRoot() ?? schema
        return try #require(JSONSerialization.jsonObject(with: encoder.encode(resolved)) as? [String: Any])
    }

    private func properties(_ schema: [String: Any]) throws -> [String: [String: Any]] {
        try #require(schema["properties"] as? [String: [String: Any]])
    }

    @Test func guidesKeepTheirValues() {
        let constant = GenerationGuide<String>.constant("fixed")
        #expect(constant.stringChoices == ["fixed"])
        #expect(constant.isConstant)

        let anyOf = GenerationGuide<String>.anyOf(["a", "b"])
        #expect(anyOf.stringChoices == ["a", "b"])
        #expect(!anyOf.isConstant)
    }

    @Test func generableGuidesReachTheSchema() throws {
        let properties = try properties(encode(StringChoiceGuided.generationSchema))
        #expect(properties["kind"]?["type"] as? String == "string")
        #expect(properties["kind"]?["const"] as? String == "fixed")
        #expect(properties["kind"]?["enum"] == nil)
        #expect(properties["described"]?["const"] as? String == "fixed")
        #expect(properties["described"]?["description"] as? String == "A described constant")
        #expect(properties["choice"]?["enum"] as? [String] == ["a", "b"])
        #expect(properties["choice"]?["const"] == nil)
        #expect(properties["maybe"]?["enum"] as? [String] == ["x", "y"])
    }

    @Test func propertyGuidesReachTheSchema() throws {
        let schema = GenerationSchema(
            type: StringChoiceGuided.self,
            properties: [
                GenerationSchema.Property(name: "kind", type: String.self, guides: [.constant("fixed")]),
                GenerationSchema.Property(name: "single", type: String.self, guides: [.anyOf(["only"])]),
            ]
        )
        let properties = try properties(encode(schema))
        #expect(properties["kind"]?["const"] as? String == "fixed")
        // A one-choice `anyOf` stays an `enum`, as in Foundation Models.
        #expect(properties["single"]?["enum"] as? [String] == ["only"])
        #expect(properties["single"]?["const"] == nil)
    }

    /// As in Foundation Models, a constant takes precedence over `anyOf` in either order,
    /// and a later guide replaces an earlier one of the same kind.
    @Test func combinedGuidesDontDependOnOrder() throws {
        let schema = GenerationSchema(
            type: StringChoiceGuided.self,
            properties: [
                GenerationSchema.Property(
                    name: "constantFirst",
                    type: String.self,
                    guides: [.constant("a"), .anyOf(["a", "b"])]
                ),
                GenerationSchema.Property(
                    name: "constantLast",
                    type: String.self,
                    guides: [.anyOf(["a", "b"]), .constant("a")]
                ),
                GenerationSchema.Property(
                    name: "twoChoices",
                    type: String.self,
                    guides: [.anyOf(["a", "b"]), .anyOf(["b", "c"])]
                ),
            ]
        )
        let properties = try properties(encode(schema))
        #expect(properties["constantFirst"]?["const"] as? String == "a")
        #expect(properties["constantFirst"]?["enum"] == nil)
        #expect(properties["constantLast"]?["const"] as? String == "a")
        #expect(properties["twoChoices"]?["enum"] as? [String] == ["b", "c"])
    }

    @Test func dynamicSchemaGuidesReachTheSchema() throws {
        let schema = try GenerationSchema(
            root: DynamicGenerationSchema(
                name: "Dynamic",
                properties: [
                    .init(name: "kind", schema: .init(type: String.self, guides: [.constant("fixed")])),
                    .init(name: "choice", schema: .init(type: String.self, guides: [.anyOf(["a", "b"])])),
                    .init(
                        name: "combined",
                        schema: .init(type: String.self, guides: [.anyOf(["a", "b"]), .constant("a")])
                    ),
                    .init(name: "plain", schema: .init(type: String.self)),
                ]
            ),
            dependencies: []
        )
        let properties = try properties(encode(schema))
        #expect(properties["kind"]?["const"] as? String == "fixed")
        #expect(properties["choice"]?["enum"] as? [String] == ["a", "b"])
        #expect(properties["combined"]?["const"] as? String == "a")
        #expect(properties["plain"]?["const"] == nil)
        #expect(properties["plain"]?["enum"] == nil)
    }

    @Test func constantsAsEnumsEncodesAOneChoiceEnum() throws {
        let properties = try properties(encode(StringChoiceGuided.generationSchema, constantsAsEnums: true))
        #expect(properties["kind"]?["type"] as? String == "string")
        #expect(properties["kind"]?["enum"] as? [String] == ["fixed"])
        #expect(properties["kind"]?["const"] == nil)
        #expect(properties["choice"]?["enum"] as? [String] == ["a", "b"])
    }

    @Test func constantsSurviveInliningAndJSONValues() throws {
        let schema = StringChoiceGuided.generationSchema
        let inlined = try JSONValue(schema.inlinedJSONSchema())
        let inlinedAsEnums = try JSONValue(schema.inlinedJSONSchema(constantsAsEnums: true))
        #expect(inlined.objectValue?["properties"]?.objectValue?["kind"]?.objectValue?["const"] == "fixed")
        #expect(inlinedAsEnums.objectValue?["properties"]?.objectValue?["kind"]?.objectValue?["enum"] == ["fixed"])

        let value = try (schema.withResolvedRoot() ?? schema).jsonValue(constantsAsEnums: true)
        #expect(value.objectValue?["properties"]?.objectValue?["kind"]?.objectValue?["enum"] == ["fixed"])
    }

    @Test func constantsDecode() throws {
        // With a `type`, as this package encodes it, and without one, as Foundation Models does.
        for json in [
            #"{"type": "object", "properties": {"kind": {"type": "string", "const": "fixed"}}, "required": ["kind"]}"#,
            #"{"type": "object", "properties": {"kind": {"const": "fixed"}}, "required": ["kind"]}"#,
        ] {
            let schema = try JSONDecoder().decode(GenerationSchema.self, from: Data(json.utf8))
            let properties = try properties(encode(schema))
            #expect(properties["kind"]?["const"] as? String == "fixed")
            #expect(properties["kind"]?["type"] as? String == "string")
        }
    }

    @Test func constantsKeepTheirPatternWhenDecoded() throws {
        for json in [
            #"{"type": "object", "properties": {"kind": {"type": "string", "const": "fixed", "pattern": "^f"}}}"#,
            #"{"type": "object", "properties": {"kind": {"const": "fixed", "pattern": "^f"}}}"#,
        ] {
            let schema = try JSONDecoder().decode(GenerationSchema.self, from: Data(json.utf8))
            let properties = try properties(encode(schema))
            #expect(properties["kind"]?["const"] as? String == "fixed")
            #expect(properties["kind"]?["pattern"] as? String == "^f")
        }
    }

    @Test func constantsAreNotEqualToOneChoiceEnums() {
        let constant = GenerationSchema.Node.string(.init(enumChoices: ["fixed"], isConstant: true))
        let oneChoice = GenerationSchema.Node.string(.init(enumChoices: ["fixed"]))
        #expect(constant != oneChoice)
    }
}
