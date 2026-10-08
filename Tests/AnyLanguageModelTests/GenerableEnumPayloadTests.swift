import Foundation
import Testing

@testable import AnyLanguageModel

@Generable
private enum Operand: Equatable {
    case name(String)
    case number(Double)
}

@Generable(description: "One thing a clause does.")
private enum Step: Equatable {
    case set(field: String, to: Operand)
    case add(Operand, to: String)
    case pair(Int, Int)
    case note(String?)
    case reset
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

    @Test func aMissingOptionalValueIsLeftOut() throws {
        #expect(try json(Step.note(nil).generatedContent.jsonString) == json(#"{"type": "note"}"#))
        #expect(try Step(GeneratedContent(json: #"{"type": "note"}"#)) == .note(nil))
        #expect(try Step(GeneratedContent(json: #"{"type": "note", "value": "hi"}"#)) == .note("hi"))
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
        #expect(properties["type"]?["enum"] as? [String] == ["set"])
        #expect(Set(set["required"] as? [String] ?? []) == ["type", "field", "to"])
        #expect(set["description"] as? String == "One thing a clause does.")

        let pair = try caseSchema("Pair")
        #expect(Set((pair["properties"] as? [String: Any] ?? [:]).keys) == ["type", "value", "value1"])

        let note = try caseSchema("Note")
        #expect(note["required"] as? [String] == ["type"])

        let reset = try caseSchema("Reset")
        #expect(Set((reset["properties"] as? [String: Any] ?? [:]).keys) == ["type"])
    }
}
