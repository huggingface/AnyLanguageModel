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
        let data = try JSONEncoder().encode(try T.generationSchema.inlinedJSONSchema())
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
}
