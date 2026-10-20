import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Decoding oneOf")
struct OneOfDecodingTests {
    /// Values described one by one, as constants under `oneOf`, are the choices of the value.
    @Test func describedValuesAreChoices() throws {
        let json = #"""
            {"type":"object","properties":{"kind":{"type":"string","oneOf":[
              {"const":"text","description":"Words to read."},
              {"const":"button","description":"A button."}
            ]}},"required":["kind"]}
            """#
        let schema = try JSONDecoder().decode(GenerationSchema.self, from: Data(json.utf8))
        guard case .object(let object) = schema.root, case .anyOf(let choices)? = object.properties["kind"] else {
            Issue.record("kind is no choice: \(schema.root)")
            return
        }
        let constants = choices.compactMap { choice -> String? in
            guard case .string(let string) = choice, string.isConstant else { return nil }
            return string.enumChoices?.first
        }
        #expect(constants == ["text", "button"])
    }
}
