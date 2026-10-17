import Testing
import AnyLanguageModel
import Foundation

@Generable
private struct TestStructWithMultilineDescription {
    @Guide(
        description: """
            This is a multi-line description.
            It spans multiple lines.
            """
    )
    var field: String
}

@Generable
private struct TestStructWithSpecialCharacters {
    @Guide(description: "A description with \"quotes\" and backslashes \\")
    var field: String
}

@Generable
private struct TestStructWithNewlines {
    @Guide(description: "Line 1\nLine 2\nLine 3")
    var field: String
}

@Generable
struct TestArguments {
    @Guide(description: "A name field")
    var name: String

    @Guide(description: "An age field")
    var age: Int
}

@Generable
private struct ArrayItem {
    @Guide(description: "A name")
    var name: String
}

@Generable
private struct ArrayContainer {
    @Guide(description: "Items", .count(2))
    var items: [ArrayItem]
}

@Generable
private struct PrimitiveContainer {
    @Guide(description: "A title")
    var title: String

    @Guide(description: "A count")
    var count: Int
}

@Generable
private enum TestEnum: Equatable {
    case optionA
    case optionB
    case optionC
}

@Generable
private struct TestNestedInner: Equatable {
    var value: String
    var count: Int
}

@Generable
private struct PrimitiveArrayContainer {
    @Guide(description: "Names", .count(2))
    var names: [String]
}

@Generable
private struct OptionalArrayContainer {
    @Guide(description: "Optional names", .count(2))
    var names: [String]?
}

@Generable
private struct NestedArrayContainer {
    @Guide(description: "Nested items", .count(2))
    var items: [[ArrayItem]]
}

@Generable
private struct OptionalPrimitiveContainer {
    @Guide(description: "Optional title")
    var title: String?

    @Guide(description: "Optional count")
    var count: Int?

    @Guide(description: "Optional flag")
    var flag: Bool?
}

@Generable
private struct OptionalItemContainer {
    @Guide(description: "Optional item")
    var item: ArrayItem?
}

@Generable
private struct OptionalItemsContainer {
    @Guide(description: "Optional items", .count(2))
    var items: [ArrayItem]?
}

@Generable
private struct TestNestedOuter: Equatable {
    var name: String
    var inner: TestNestedInner
}

@Generable
private struct TestStructWithEnum: Equatable {
    var label: String
    var choice: TestEnum
}

@Generable
private struct TestStructWithArray: Equatable {
    var title: String
    @Guide(.count(3))
    var items: [String]
}

@Generable
struct CodableItem: Codable, Equatable {
    var name: String
}

@Suite("Generable Macro")
struct GenerableMacroTests {
    @Test("@Guide description with multiline string")
    func multilineGuideDescription() async throws {
        let schema = TestStructWithMultilineDescription.generationSchema
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(schema)

        // Verify that the schema can be encoded without errors (no unterminated strings)
        #expect(jsonData.count > 0)

        // Verify it can be decoded back
        let decoder = JSONDecoder()
        let decodedSchema = try decoder.decode(GenerationSchema.self, from: jsonData)
        #expect(decodedSchema.debugDescription.contains("object"))
    }

    @Test("@Guide description with special characters")
    func guideDescriptionWithSpecialCharacters() async throws {
        let schema = TestStructWithSpecialCharacters.generationSchema
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(schema)
        let jsonString = String(data: jsonData, encoding: .utf8)!

        // Verify the special characters are escaped
        #expect(jsonString.contains(#"\\\"quotes\\\""#))
        #expect(jsonString.contains(#"backslashes \\\\"#))

        // Verify roundtrip encoding/decoding works
        let decoder = JSONDecoder()
        let decodedSchema = try decoder.decode(GenerationSchema.self, from: jsonData)
        #expect(decodedSchema.debugDescription.contains("object"))
    }

    @Test("@Guide description with newlines")
    func guideDescriptionWithNewlines() async throws {
        let schema = TestStructWithNewlines.generationSchema
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(schema)

        // Verify that the schema can be encoded without errors
        #expect(jsonData.count > 0)

        // Verify roundtrip encoding/decoding works
        let decoder = JSONDecoder()
        let decodedSchema = try decoder.decode(GenerationSchema.self, from: jsonData)
        #expect(decodedSchema.debugDescription.contains("object"))
    }

    @MainActor
    @Generable
    struct MainActorIsolatedStruct {
        @Guide(description: "A test field")
        var field: String
    }

    @MainActor
    @Test("@MainActor isolation")
    func mainActorIsolation() async throws {
        let generatedContent = GeneratedContent(properties: [
            "field": "test value"
        ])
        let instance = try MainActorIsolatedStruct(generatedContent)
        #expect(instance.field == "test value")

        let convertedBack = instance.generatedContent
        let decoded = try MainActorIsolatedStruct(convertedBack)
        #expect(decoded.field == "test value")

        let schema = MainActorIsolatedStruct.generationSchema
        #expect(schema.debugDescription.contains("MainActorIsolatedStruct"))

        let partiallyGenerated = instance.asPartiallyGenerated()
        #expect(partiallyGenerated.field == "test value")
    }

    @Test("Memberwise initializer")
    func memberwiseInit() throws {
        // This is the natural Swift way to create instances
        let args = TestArguments(name: "Alice", age: 30)

        #expect(args.name == "Alice")
        #expect(args.age == 30)

        // The generatedContent should also be properly populated
        let content = args.generatedContent
        #expect(content.jsonString.contains("Alice"))
        #expect(content.jsonString.contains("30"))
    }

    @Test("Create instance from GeneratedContent")
    func fromGeneratedContent() throws {
        let generationID = GenerationID()
        let content = GeneratedContent(
            properties: [
                "name": GeneratedContent("Bob"),
                "age": GeneratedContent(kind: .number(25)),
            ],
            id: generationID
        )

        let args = try TestArguments(content)
        #expect(args.name == "Bob")
        #expect(args.age == 25)
        // As in Foundation Models, the value keeps only its properties, not the generation's id.
        let partial = args.asPartiallyGenerated()
        #expect(partial.name == "Bob")
        #expect(partial.age == 25)
    }

    /// A value that's also `Codable` encodes only its own properties: nothing generated rides
    /// along with it.
    @Test("Codable encodes only the declared properties")
    func codableEncodesOnlyDeclaredProperties() throws {
        let value = try CodableItem(GeneratedContent(properties: ["name": GeneratedContent("Bob")], id: GenerationID()))
        let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any]
        #expect(json?.keys.sorted() == ["name"])
        #expect(value == CodableItem(name: "Bob"))
    }

    @Test("Array properties use partially generated element types")
    func arrayPropertiesUsePartiallyGeneratedElements() throws {
        let content = GeneratedContent(
            properties: [
                "items": GeneratedContent(
                    kind: .array([
                        GeneratedContent(properties: ["name": "Alpha"]),
                        GeneratedContent(properties: ["name": "Beta"]),
                    ])
                )
            ]
        )

        let container = try ArrayContainer(content)
        let partial = container.asPartiallyGenerated()
        #expect(partial.items?.count == 2)
        #expect(partial.items?.first?.name == "Alpha")
    }

    @Test("Primitive properties use concrete partial types")
    func primitivePropertiesRemainUnchanged() throws {
        let content = GeneratedContent(
            properties: [
                "title": "Hello",
                "count": 3,
            ]
        )

        let container = try PrimitiveContainer(content)
        let partial = container.asPartiallyGenerated()
        #expect(partial.title == "Hello")
        #expect(partial.count == 3)
    }

    @Test("Array primitives use concrete element types")
    func arrayPrimitivesRemainConcrete() throws {
        let content = GeneratedContent(
            properties: [
                "names": GeneratedContent(
                    kind: .array([
                        GeneratedContent("Alpha"),
                        GeneratedContent("Beta"),
                    ])
                )
            ]
        )

        let container = try PrimitiveArrayContainer(content)
        let partial = container.asPartiallyGenerated()
        #expect(partial.names?.count == 2)
        #expect(partial.names?.first == "Alpha")
    }

    @Test("Optional primitive arrays remain concrete")
    func optionalPrimitiveArraysRemainConcrete() throws {
        let content = GeneratedContent(
            properties: [
                "names": GeneratedContent(
                    kind: .array([
                        GeneratedContent("Alpha"),
                        GeneratedContent("Beta"),
                    ])
                )
            ]
        )

        let container = try OptionalArrayContainer(content)
        let partial = container.asPartiallyGenerated()
        #expect(partial.names?.count == 2)
        #expect(partial.names?.first == "Alpha")
    }

    @Test("Nested arrays of generable types are handled")
    func nestedArraysGenerateNestedPartialTypes() throws {
        let content = GeneratedContent(
            properties: [
                "items": GeneratedContent(
                    kind: .array([
                        GeneratedContent(
                            kind: .array([
                                GeneratedContent(properties: ["name": "Alpha"]),
                                GeneratedContent(properties: ["name": "Beta"]),
                            ])
                        ),
                        GeneratedContent(
                            kind: .array([
                                GeneratedContent(properties: ["name": "Gamma"])
                            ])
                        ),
                    ])
                )
            ]
        )

        let container = try NestedArrayContainer(content)
        let partial = container.asPartiallyGenerated()
        #expect(partial.items?.count == 2)
        #expect(partial.items?.first?.count == 2)
        #expect(partial.items?.first?.first?.name == "Alpha")
        #expect(partial.items?.last?.first?.name == "Gamma")
    }

    @Test("Optional primitive properties are handled")
    func optionalPrimitivePropertiesHandled() throws {
        let content = GeneratedContent(
            properties: [
                "title": "Hello",
                "count": 3,
                "flag": true,
            ]
        )

        let container = try OptionalPrimitiveContainer(content)
        let partial = container.asPartiallyGenerated()
        #expect(partial.title == "Hello")
        #expect(partial.count == 3)
        #expect(partial.flag == true)
    }

    @Test("Optional generable properties are handled")
    func optionalGenerableItemBecomesPartial() throws {
        let content = GeneratedContent(
            properties: [
                "item": GeneratedContent(properties: ["name": "Alpha"])
            ]
        )

        let container = try OptionalItemContainer(content)
        let partial = container.asPartiallyGenerated()
        #expect(partial.item?.name == "Alpha")
    }

    @Test("Optional arrays of generable types are handled")
    func optionalGenerableArraysTransformToPartialArrays() throws {
        let content = GeneratedContent(
            properties: [
                "items": GeneratedContent(
                    kind: .array([
                        GeneratedContent(properties: ["name": "Alpha"]),
                        GeneratedContent(properties: ["name": "Beta"]),
                    ])
                )
            ]
        )

        let container = try OptionalItemsContainer(content)
        let partial = container.asPartiallyGenerated()
        #expect(partial.items?.count == 2)
        #expect(partial.items?.first?.name == "Alpha")
    }

    @Test("Missing optional properties become nil in partials")
    func missingOptionalProperties() throws {
        let content = GeneratedContent(properties: [:])

        let primitive = try OptionalPrimitiveContainer(content).asPartiallyGenerated()
        #expect(primitive.title == nil)
        #expect(primitive.count == nil)
        #expect(primitive.flag == nil)

        let item = try OptionalItemContainer(content).asPartiallyGenerated()
        #expect(item.item == nil)

        let items = try OptionalItemsContainer(content).asPartiallyGenerated()
        #expect(items.items == nil)

        let names = try OptionalArrayContainer(content).asPartiallyGenerated()
        #expect(names.names == nil)
    }

    @Test("Schema generation includes optional properties")
    func schemaIncludesOptionalProperties() throws {
        let schema = OptionalPrimitiveContainer.generationSchema
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(schema)
        let jsonString = String(data: jsonData, encoding: .utf8) ?? ""

        #expect(jsonString.contains("\"title\""))
        #expect(jsonString.contains("\"count\""))
        #expect(jsonString.contains("\"flag\""))
    }

    @Test("Enum round-trip conversion")
    func enumRoundTrip() throws {
        for choice in [TestEnum.optionA, TestEnum.optionB, TestEnum.optionC] {
            let restored = try TestEnum(choice.generatedContent)
            #expect(choice == restored)
        }
    }

    @Test("Nested struct round-trip conversion")
    func nestedStructRoundTrip() throws {
        let original = TestNestedOuter(
            name: "outer",
            inner: TestNestedInner(value: "inner", count: 42)
        )
        let restored = try TestNestedOuter(original.generatedContent)
        #expect(original == restored)
    }

    @Test("Struct with enum round-trip conversion")
    func structWithEnumRoundTrip() throws {
        let original = TestStructWithEnum(label: "test", choice: .optionB)
        let restored = try TestStructWithEnum(original.generatedContent)
        #expect(original == restored)
    }

    @Test("Struct with array round-trip conversion")
    func structWithArrayRoundTrip() throws {
        let original = TestStructWithArray(title: "list", items: ["a", "b", "c"])
        let restored = try TestStructWithArray(original.generatedContent)
        #expect(original == restored)
    }
}

// MARK: - #Playground Usage

// The `#Playground` macro doesn't see the memberwise initializer
// that `@Generable` expands. This is a limitation of how macros compose:
// one macro's expansion isn't visible within another macro's body.
//
// The following code demonstrates workarounds for this limitation.

#if canImport(Playgrounds)
    import Playgrounds

    // Use the `GeneratedContent` initializer explicitly.
    #Playground {
        let content = GeneratedContent(properties: [
            "name": "Alice",
            "age": 30,
        ])
        let _ = try TestArguments(content)
    }

    // Define a factory method as an alternative to the memberwise initializer.
    extension TestArguments {
        static func create(name: String, age: Int) -> TestArguments {
            try! TestArguments(
                GeneratedContent(properties: [
                    "name": name,
                    "age": age,
                ])
            )
        }
    }

    #Playground {
        let _ = TestArguments.create(name: "Bob", age: 42)
    }
#endif  // canImport(Playgrounds)
