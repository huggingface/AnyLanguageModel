import Foundation
import Testing

@testable import AnyLanguageModel

@Generable
private struct MissingPropertyContact {
    var name: String
    var age: Int
    var tags: [String]
    var nickname: String?
}

@Generable
private struct MissingPropertyOuter {
    var inner: MissingPropertyInner
}

@Generable
private struct MissingPropertyInner {
    var value: Int
}

/// As in Foundation Models, a required property that's missing is an error, not a default value,
/// and an optional one that's missing or `null` is `nil`.
@Suite("Missing Generable properties")
struct GenerableMissingPropertyTests {
    @Test(arguments: [
        (#"{}"#, "name"),
        (#"{"name": "Ann"}"#, "age"),
        (#"{"name": "Ann", "age": 30}"#, "tags"),
    ])
    func missingRequiredPropertyThrows(json: String, missing: String) throws {
        let content = try GeneratedContent(json: json)
        #expect(throws: GeneratedContentError.propertyNotFound(missing)) {
            try MissingPropertyContact(content)
        }
    }

    @Test(arguments: [
        #"{"name": "Ann", "age": 30, "tags": []}"#, #"{"name": "Ann", "age": 30, "tags": [], "nickname": null}"#,
    ])
    func missingOrNullOptionalPropertyIsNil(json: String) throws {
        let contact = try MissingPropertyContact(GeneratedContent(json: json))
        #expect(contact.name == "Ann")
        #expect(contact.age == 30)
        #expect(contact.tags.isEmpty)
        #expect(contact.nickname == nil)
    }

    @Test func missingNestedPropertyThrows() throws {
        #expect(throws: GeneratedContentError.propertyNotFound("inner")) {
            try MissingPropertyOuter(GeneratedContent(json: #"{}"#))
        }
        #expect(throws: GeneratedContentError.propertyNotFound("value")) {
            try MissingPropertyOuter(GeneratedContent(json: #"{"inner": {}}"#))
        }
    }

    /// A partially generated value is still made from whatever has been generated so far.
    @Test func partiallyGeneratedValueTakesMissingProperties() throws {
        let partial = try MissingPropertyContact.PartiallyGenerated(GeneratedContent(json: #"{"name": "Ann"}"#))
        #expect(partial.name == "Ann")
        #expect(partial.age == nil)
        #expect(partial.tags == nil)
    }
}
