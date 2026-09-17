import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("GeneratedContent JSON coding")
struct GeneratedContentJSONTests {
    @Generable
    struct NovelIdea {
        let title: String
        let pages: Int
        let tags: [String]
    }

    // MARK: - init(json: Data)

    @Test func dataInitializerParsesCompleteJSON() throws {
        let data = Data(#"{"title": "Dune", "pages": 412, "tags": ["sci-fi", "classic"]}"#.utf8)
        let content = try GeneratedContent(json: data)
        let idea = try NovelIdea(content)
        #expect(idea.title == "Dune")
        #expect(idea.pages == 412)
        #expect(idea.tags == ["sci-fi", "classic"])
        #expect(content.id == nil)
    }

    @Test func dataInitializerMatchesStringInitializer() throws {
        let json = #"{"a": 1, "b": [true, null, "x"], "c": {"d": 2.5}}"#
        let fromData = try GeneratedContent(json: Data(json.utf8))
        let fromString = try GeneratedContent(json: json)
        #expect(fromData.jsonValue == fromString.jsonValue)
        #expect(fromData.jsonValue == ["a": 1.0, "b": [true, nil, "x"], "c": ["d": 2.5]])
    }

    @Test func dataInitializerParsesFragments() throws {
        #expect(try GeneratedContent(json: Data("42".utf8)).kind == .number(42))
        #expect(try GeneratedContent(json: Data("true".utf8)).kind == .bool(true))
        #expect(try GeneratedContent(json: Data("null".utf8)).kind == .null)
        #expect(try GeneratedContent(json: Data(#""hi""#.utf8)).kind == .string("hi"))
    }

    @Test func dataInitializerCompletesPartialJSON() throws {
        let partial = Data(#"{"title": "A story of"#.utf8)
        let content = try GeneratedContent(json: partial)
        #expect(try content.value(String.self, forProperty: "title") == "A story of")

        let partialArray = Data(#"[1, 2, 3"#.utf8)
        #expect(try GeneratedContent(json: partialArray).kind == .array([1, 2, 3].map { GeneratedContent($0) }))
    }

    @Test(arguments: [
        (#"{"a": {"b": "x"#, #"{"a": {"b": "x"}}"#),
        (#"{"a": [1, 2"#, #"{"a": [1, 2]}"#),
        (#"[{"a": 1}, {"b": "#, #"[{"a": 1}, {"b": null}]"#),
        (#"{"a": 1, "b": tr"#, #"{"a": 1, "b": true}"#),
        (#"{"a": 12."#, #"{"a": 12.0}"#),
        (#"{"a": "esc\"#, #"{"a": "esc"}"#),
        (#"{"a": "q\" more\u00"#, #"{"a": "q\" more"}"#),
        (#"{"na"#, #"{"na": null}"#),
    ])
    func completesTruncatedStreamingJSON(partial: String, expected: String) throws {
        let fromString = try GeneratedContent(json: partial)
        let fromData = try GeneratedContent(json: Data(partial.utf8))
        let expectedContent = try GeneratedContent(json: expected)
        #expect(fromString.jsonValue == expectedContent.jsonValue)
        #expect(fromData.jsonValue == expectedContent.jsonValue)
    }

    @Test func partialGenerableDecodesFromTruncatedJSON() throws {
        let partial = #"{"title": "Dune", "pages": 41"#
        let idea = try NovelIdea.PartiallyGenerated(GeneratedContent(json: partial))
        #expect(idea.title == "Dune")
        #expect(idea.pages == 41)
        #expect(idea.tags == nil)
    }

    @Test func dataInitializerFallsBackToString() throws {
        let content = try GeneratedContent(json: Data("  not json  ".utf8))
        #expect(content.kind == .string("not json"))
    }

    // MARK: - jsonData

    @Test func jsonDataRoundTrips() throws {
        let content = GeneratedContent(properties: [
            "name": "Johnny Appleseed",
            "age": 30,
            "active": true,
            "scores": [1.5, 2],
            "nothing": GeneratedContent(kind: .null),
        ])

        let data = content.jsonData
        #expect(try GeneratedContent(json: content.jsonString).jsonValue == GeneratedContent(json: data).jsonValue)
        #expect(try GeneratedContent(json: data).jsonValue == content.jsonValue)

        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(object?["name"] as? String == "Johnny Appleseed")
        #expect(object?["age"] as? Int == 30)
        #expect(object?["active"] as? Bool == true)
    }

    // MARK: - JSONValue bridging

    @Test func initFromJSONValue() throws {
        let value: JSONValue = [
            "title": "Dune",
            "pages": 412,
            "ratio": 1.5,
            "tags": ["sci-fi", "classic"],
            "meta": ["ok": true, "none": nil],
        ]
        let content = GeneratedContent(value)
        let idea = try NovelIdea(content)
        #expect(idea.title == "Dune")
        #expect(idea.pages == 412)
        #expect(idea.tags == ["sci-fi", "classic"])
        #expect(try content.value(Double.self, forProperty: "ratio") == 1.5)

        let meta = try content.value(GeneratedContent.self, forProperty: "meta")
        #expect(try meta.value(Bool.self, forProperty: "ok") == true)
        #expect(try meta.value(GeneratedContent.self, forProperty: "none").kind == .null)

        let id = GenerationID()
        #expect(GeneratedContent(value, id: id).id == id)
    }

    @Test func jsonValueRoundTrips() throws {
        let value: JSONValue = [
            "title": "Dune",
            "pages": 412.0,
            "tags": ["sci-fi", "classic"],
            "meta": ["ok": true, "none": nil],
        ]
        #expect(GeneratedContent(value).jsonValue == value)

        let content = try GeneratedContent(json: #"{"a": [1, {"b": null}], "c": "d"}"#)
        #expect(content.jsonValue == ["a": [1.0, ["b": nil]], "c": "d"])
        #expect(GeneratedContent(content.jsonValue).jsonValue == content.jsonValue)
    }

    @Test func jsonValueIsConvertible() throws {
        let value: JSONValue = ["x": 1.0]
        #expect(value.generatedContent == GeneratedContent(value))
        #expect(GeneratedContent(value).jsonValue == value)
        #expect(try JSONValue(GeneratedContent(value)) == value)
        #expect(try GeneratedContent(value).value(JSONValue.self) == value)
        #expect(try GeneratedContent(value).value(JSONValue.self, forProperty: "x") == 1.0)
    }

    // MARK: - Codable

    @Test func codableRoundTripPreservesIDAndOrder() throws {
        let id = GenerationID()
        let content = GeneratedContent(
            kind: .structure(
                properties: ["z": GeneratedContent(1), "a": GeneratedContent("two")],
                orderedKeys: ["z", "a"]
            ),
            id: id
        )
        let data = try JSONEncoder().encode(content)
        let decoded = try JSONDecoder().decode(GeneratedContent.self, from: data)
        #expect(decoded == content)
        #expect(decoded.id == id)
        #expect(decoded.jsonValue == ["z": 1.0, "a": "two"])
    }

    @Test func decodesPlainJSONObject() throws {
        struct ProviderResponse: Decodable {
            let content: GeneratedContent
            let model: String
        }

        let json = #"{"model": "test", "content": {"title": "Dune", "pages": 412, "tags": ["sci-fi"]}}"#
        let response = try JSONDecoder().decode(ProviderResponse.self, from: Data(json.utf8))
        #expect(response.model == "test")
        #expect(response.content.id == nil)
        let idea = try NovelIdea(response.content)
        #expect(idea.title == "Dune")
        #expect(idea.pages == 412)
        #expect(idea.tags == ["sci-fi"])
    }

    @Test func decodesPlainJSONFragments() throws {
        let decoder = JSONDecoder()
        #expect(try decoder.decode(GeneratedContent.self, from: Data("42".utf8)).kind == .number(42))
        #expect(try decoder.decode(GeneratedContent.self, from: Data("false".utf8)).kind == .bool(false))
        #expect(try decoder.decode(GeneratedContent.self, from: Data("null".utf8)).kind == .null)
        #expect(try decoder.decode(GeneratedContent.self, from: Data(#""s""#.utf8)).kind == .string("s"))
        #expect(
            try decoder.decode(GeneratedContent.self, from: Data("[1, \"a\"]".utf8)).kind
                == .array([GeneratedContent(1), GeneratedContent("a")])
        )
        #expect(try decoder.decode([GeneratedContent].self, from: Data("[{}, []]".utf8)).count == 2)
    }

    @Test func decodesPlainJSONObjectWithKindKey() throws {
        // A plain object whose "kind" value is not a canonical Kind should decode as plain JSON.
        let json = #"{"kind": "ordinary parameter", "city": "Paris"}"#
        let content = try JSONDecoder().decode(GeneratedContent.self, from: Data(json.utf8))
        #expect(try content.value(String.self, forProperty: "kind") == "ordinary parameter")
        #expect(try content.value(String.self, forProperty: "city") == "Paris")
    }

    @Test func decodesPlainJSONObjectWithCanonicalLookingKindAndSiblings() throws {
        let json = #"{"kind": {"type": "string", "value": "inner"}, "sibling": "kept"}"#
        let content = try JSONDecoder().decode(GeneratedContent.self, from: Data(json.utf8))
        #expect(try content.value(String.self, forProperty: "sibling") == "kept")
        let kind = try content.value(GeneratedContent.self, forProperty: "kind")
        #expect(try kind.value(String.self, forProperty: "type") == "string")
        #expect(try kind.value(String.self, forProperty: "value") == "inner")
    }

    @Test func jsonValueOmitsPropertiesAbsentFromOrderedKeys() throws {
        let content = GeneratedContent(
            kind: .structure(
                properties: ["a": GeneratedContent(1), "hidden": GeneratedContent(2)],
                orderedKeys: ["a"]
            )
        )
        #expect(content.jsonValue == ["a": 1.0])
        #expect(try GeneratedContent(json: content.jsonData).jsonValue == content.jsonValue)
    }

    @Test func encodedFormIsNotPlainJSON() throws {
        let content = GeneratedContent(properties: ["a": 1])
        let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(content)) as? [String: Any]
        #expect(object?["kind"] != nil)
        #expect(object?["a"] == nil)
    }
}
