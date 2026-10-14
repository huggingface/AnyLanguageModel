import Foundation
import Testing

@testable import AnyLanguageModel

// `GenerableEnumPayloadFoundationModelsTests` compares these enums
// with the same enums in Foundation Models.

@Generable
enum Operand: Equatable {
    case name(String)
    case number(Double)
}

@Generable(description: "One thing a clause does.")
enum Step: Equatable {
    case set(field: String, to: Operand)
    case add(Operand, to: String)
    case pair(Int, Int)
    case note(String?)
    case reset
}

typealias MaybeText = String?

/// Optional values, with each spelling of an optional type.
@Generable
enum Annotation: Equatable {
    case note(String?)
    case alias(MaybeText)
    case generic(Optional<Int>)
    case qualified(Swift.Optional<String>)
}

@Generable
private enum Casing: Equatable {
    case foo(Int)
    case Foo(String)
}

@Generable
private enum Shelf: Equatable {
    @Generable
    struct DiscriminatedItem: Equatable {
        var title: String
    }

    case item(DiscriminatedItem)
}

/// Enums with associated values are generated the way Foundation Models generates them: each
/// case an object whose `type` names it, with its values as properties. The expected values are
/// what Foundation Models' own `@Generable` produces for the same enums.
@Suite("Generable enums with associated values")
struct GenerableEnumPayloadTests {
    private func json(_ text: String) throws -> NSDictionary {
        try #require(JSONSerialization.jsonObject(with: Data(text.utf8)) as? NSDictionary)
    }

    @Test func casesAreObjectsNamedByType() throws {
        let expected: [(Step, String)] = [
            (
                .set(field: "count", to: .number(1)),
                #"{"type": "set", "field": "count", "to": {"type": "number", "value": 1}}"#
            ),
            (
                .add(.name("amount"), to: "count"),
                #"{"type": "add", "value": {"type": "name", "value": "amount"}, "to": "count"}"#
            ),
            (.pair(1, 2), #"{"type": "pair", "value": 1, "value1": 2}"#),
            (.reset, #"{"type": "reset"}"#),
        ]
        for (step, content) in expected {
            #expect(try json(step.generatedContent.jsonString) == json(content))
            #expect(try Step(GeneratedContent(json: content)) == step)
        }
    }

    @Test func typeComesFirst() throws {
        guard case .structure(_, let keys) = Step.set(field: "count", to: .number(1)).generatedContent.kind else {
            Issue.record("Expected a structure")
            return
        }
        #expect(keys == ["type", "field", "to"])
    }

    @Test func aMissingOptionalValueIsNil() throws {
        #expect(try Step(GeneratedContent(json: #"{"type": "note"}"#)) == .note(nil))
        #expect(try Step(GeneratedContent(json: #"{"type": "note", "value": "hi"}"#)) == .note("hi"))
    }

    /// Foundation Models writes a `nil` value as `null`, and reads `null` as `nil`.
    @Test func aNullOptionalValueIsNil() throws {
        #expect(try Step(GeneratedContent(json: #"{"type": "note", "value": null}"#)) == .note(nil))
    }

    /// The macro can't see through a type alias,
    /// so nothing it generates depends on how an optional type is spelled.
    @Test func optionalValuesDoNotDependOnSpelling() throws {
        let expected: [(Annotation, String)] = [
            (.note(nil), #"{"type":"note","value":null}"#),
            (.alias(nil), #"{"type":"alias","value":null}"#),
            (.generic(nil), #"{"type":"generic","value":null}"#),
            (.qualified(nil), #"{"type":"qualified","value":null}"#),
            (.note("a"), #"{"type":"note","value":"a"}"#),
            (.alias("b"), #"{"type":"alias","value":"b"}"#),
            (.generic(2), #"{"type":"generic","value":2}"#),
            (.qualified("c"), #"{"type":"qualified","value":"c"}"#),
        ]
        for (annotation, content) in expected {
            #expect(annotation.generatedContent.jsonString == content)
            #expect(try Annotation(GeneratedContent(json: content)) == annotation)
        }

        let missing: [(String, Annotation)] = [
            ("note", .note(nil)), ("alias", .alias(nil)), ("generic", .generic(nil)), ("qualified", .qualified(nil)),
        ]
        for (type, annotation) in missing {
            #expect(try Annotation(GeneratedContent(json: #"{"type":"\#(type)"}"#)) == annotation)
        }

        let (defs, choices) = try caseDefinitions(Annotation.generationSchema)
        #expect(choices.count == 4)
        for name in choices {
            #expect(defs[name]?["required"] as? [String] == ["type"])
        }
    }

    @Test func anUnknownTypeIsAnError() {
        #expect(throws: DecodingError.self) {
            try Step(GeneratedContent(json: #"{"type": "jump"}"#))
        }
    }

    @Test func schemaHasAnObjectForEachCase() throws {
        let encoded = try JSONEncoder().encode(Step.generationSchema)
        let schema = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let defs = try #require(schema["$defs"] as? [String: [String: Any]])
        func caseSchema(_ name: String) throws -> [String: Any] {
            try #require(defs.first { $0.key.hasSuffix(".Discriminated\(name)") }?.value)
        }

        let root = try #require(defs.first { $0.key.hasSuffix(".Step") }?.value)
        #expect((root["anyOf"] as? [Any])?.count == 5)

        let set = try caseSchema("Set")
        let properties = try #require(set["properties"] as? [String: [String: Any]])
        #expect(Set(properties.keys) == ["type", "field", "to"])
        #expect(properties["type"]?["const"] as? String == "set")
        #expect(Set(set["required"] as? [String] ?? []) == ["type", "field", "to"])
        #expect(set["description"] as? String == "One thing a clause does.")

        let pair = try caseSchema("Pair")
        #expect(Set((pair["properties"] as? [String: Any] ?? [:]).keys) == ["type", "value", "value1"])

        let note = try caseSchema("Note")
        #expect(note["required"] as? [String] == ["type"])

        let reset = try caseSchema("Reset")
        #expect(Set((reset["properties"] as? [String: Any] ?? [:]).keys) == ["type"])
    }

    /// Like Foundation Models, a case's `type` is a constant,
    /// which providers without `const` get as a one-choice `enum`.
    @Test func typeIsAConstant() throws {
        let inlined = try JSONValue(Step.generationSchema.inlinedJSONSchema())
        let asEnum = try JSONValue(Step.generationSchema.inlinedJSONSchema(constantsAsEnums: true))
        guard case .array(let cases)? = inlined.objectValue?["anyOf"],
            case .array(let enumCases)? = asEnum.objectValue?["anyOf"]
        else {
            Issue.record("Expected a union of cases")
            return
        }
        let type = cases.first?.objectValue?["properties"]?.objectValue?["type"]
        #expect(type == ["type": "string", "const": "set"])
        let enumType = enumCases.first?.objectValue?["properties"]?.objectValue?["type"]
        #expect(enumType == ["type": "string", "enum": ["set"]])
    }

    /// The encoded definitions of a schema, and the definitions its root's choices refer to.
    private func caseDefinitions(
        _ schema: GenerationSchema
    ) throws -> (defs: [String: [String: Any]], choices: [String]) {
        let encoded = try JSONEncoder().encode(schema)
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let defs = try #require(object["$defs"] as? [String: [String: Any]])
        let rootName = try #require((object["$ref"] as? String)?.replacingOccurrences(of: "#/$defs/", with: ""))
        let anyOf = try #require(defs[rootName]?["anyOf"] as? [[String: String]])
        let choices = anyOf.compactMap { $0["$ref"]?.replacingOccurrences(of: "#/$defs/", with: "") }
        return (defs, choices)
    }

    @Test func casesDifferingInCaseHaveTheirOwnSchemas() throws {
        let (defs, choices) = try caseDefinitions(Casing.generationSchema)

        #expect(choices.count == 2)
        #expect(Set(choices).count == 2)
        #expect(choices.first?.hasSuffix(".DiscriminatedFoo") == true)
        #expect(choices.last?.hasSuffix(".DiscriminatedFoo2") == true)
        let types = choices.map { name in
            (defs[name]?["properties"] as? [String: [String: Any]])?["type"]?["const"] as? String
        }
        #expect(types == ["foo", "Foo"])
        let values = choices.map { name in
            (defs[name]?["properties"] as? [String: [String: Any]])?["value"]?["type"] as? String
        }
        #expect(values == ["integer", "string"])

        // The names depend only on the declaration.
        #expect(Casing.generationSchema == Casing.generationSchema)

        let expected: [(Casing, String)] = [
            (.foo(1), #"{"type":"foo","value":1}"#),
            (.Foo("x"), #"{"type":"Foo","value":"x"}"#),
        ]
        for (casing, content) in expected {
            #expect(casing.generatedContent.jsonString == content)
            #expect(try Casing(GeneratedContent(json: content)) == casing)
        }
    }

    @Test func aCaseDoesNotTakeTheNameOfAValueType() throws {
        let (defs, choices) = try caseDefinitions(Shelf.generationSchema)
        let itemName = String(reflecting: Shelf.DiscriminatedItem.self)

        let caseName = try #require(choices.first)
        #expect(caseName == "\(itemName)2")
        let caseProperties = try #require(defs[caseName]?["properties"] as? [String: [String: Any]])
        #expect(caseProperties["value"]?["$ref"] as? String == "#/$defs/\(itemName)")
        let itemProperties = try #require(defs[itemName]?["properties"] as? [String: Any])
        #expect(Set(itemProperties.keys) == ["title"])

        let shelf = Shelf.item(.init(title: "Dune"))
        #expect(try Shelf(GeneratedContent(json: shelf.generatedContent.jsonString)) == shelf)
    }
}
