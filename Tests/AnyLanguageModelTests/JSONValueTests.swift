import Foundation
import Testing

import enum JSONSchema.JSONValue

@testable import AnyLanguageModel

@Suite("JSONValue")
struct JSONValueTests {
    private let sample: AnyLanguageModel.JSONValue = .object([
        "null": .null,
        "bool": .bool(true),
        "int": .int(42),
        "double": .double(2.5),
        "string": .string("hello"),
        "array": .array([.int(1), .string("two"), .null]),
        "object": .object(["nested": .bool(false)]),
    ])

    // MARK: - Codable

    @Test func roundTripsThroughJSON() throws {
        let data = try JSONEncoder().encode(sample)
        let decoded = try JSONDecoder().decode(AnyLanguageModel.JSONValue.self, from: data)
        #expect(decoded == sample)
    }

    @Test func encodesAsPlainJSON() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let value: AnyLanguageModel.JSONValue = ["a": [true, nil, 1, 1.5, "x"], "b": [:]]
        let json = String(decoding: try encoder.encode(value), as: UTF8.self)
        #expect(json == #"{"a":[true,null,1,1.5,"x"],"b":{}}"#)
    }

    @Test func decodesPlainJSON() throws {
        let json = #"{"a": [true, null, 1, 1.5, "x"], "b": {"c": []}}"#
        let decoded = try JSONDecoder().decode(AnyLanguageModel.JSONValue.self, from: Data(json.utf8))
        #expect(
            decoded
                == .object([
                    "a": .array([.bool(true), .null, .int(1), .double(1.5), .string("x")]),
                    "b": .object(["c": .array([])]),
                ])
        )
    }

    @Test func decodesFragments() throws {
        let decoder = JSONDecoder()
        #expect(try decoder.decode(AnyLanguageModel.JSONValue.self, from: Data("null".utf8)) == .null)
        #expect(try decoder.decode(AnyLanguageModel.JSONValue.self, from: Data("false".utf8)) == .bool(false))
        #expect(try decoder.decode(AnyLanguageModel.JSONValue.self, from: Data("-7".utf8)) == .int(-7))
        #expect(try decoder.decode(AnyLanguageModel.JSONValue.self, from: Data("0.25".utf8)) == .double(0.25))
        #expect(try decoder.decode(AnyLanguageModel.JSONValue.self, from: Data(#""hi""#.utf8)) == .string("hi"))
    }

    @Test func matchesJSONSchemaWireFormat() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let ours = try encoder.encode(sample)
        let theirs = try encoder.encode(sample.jsonSchemaValue)
        #expect(ours == theirs)
    }

    // MARK: - Literals

    @Test func literals() {
        let null: AnyLanguageModel.JSONValue = nil
        let bool: AnyLanguageModel.JSONValue = true
        let int: AnyLanguageModel.JSONValue = 42
        let double: AnyLanguageModel.JSONValue = 2.5
        let string: AnyLanguageModel.JSONValue = "hello"
        let array: AnyLanguageModel.JSONValue = [1, "two", nil]
        let object: AnyLanguageModel.JSONValue = ["nested": false]

        #expect(null == .null)
        #expect(bool == .bool(true))
        #expect(int == .int(42))
        #expect(double == .double(2.5))
        #expect(string == .string("hello"))
        #expect(array == .array([.int(1), .string("two"), .null]))
        #expect(object == .object(["nested": .bool(false)]))
    }

    @Test func dictionaryLiteralKeepsLastDuplicateKey() {
        let value: AnyLanguageModel.JSONValue = ["key": 1, "key": 2]
        #expect(value == .object(["key": .int(2)]))
    }

    // MARK: - Initializers and accessors

    @Test func initFromEncodable() throws {
        struct Payload: Encodable {
            let name: String
            let count: Int
            let tags: [String]
        }
        let value = try AnyLanguageModel.JSONValue(Payload(name: "a", count: 3, tags: ["x"]))
        #expect(value == ["name": "a", "count": 3, "tags": ["x"]])
    }

    @Test func initFromEncodableReturnsJSONValueUnchanged() throws {
        #expect(try AnyLanguageModel.JSONValue(sample) == sample)
    }

    @Test func isNull() {
        #expect(AnyLanguageModel.JSONValue.null.isNull)
        #expect(!AnyLanguageModel.JSONValue.bool(false).isNull)
        #expect(!AnyLanguageModel.JSONValue.string("").isNull)
    }

    @Test func boolValue() {
        #expect(AnyLanguageModel.JSONValue.bool(true).boolValue == true)
        #expect(AnyLanguageModel.JSONValue.bool(false).boolValue == false)
        #expect(AnyLanguageModel.JSONValue.int(1).boolValue == nil)
        #expect(AnyLanguageModel.JSONValue.string("true").boolValue == nil)
    }

    @Test func intValue() {
        #expect(AnyLanguageModel.JSONValue.int(42).intValue == 42)
        #expect(AnyLanguageModel.JSONValue.double(42.0).intValue == nil)
        #expect(AnyLanguageModel.JSONValue.string("42").intValue == nil)
    }

    @Test func doubleValue() {
        #expect(AnyLanguageModel.JSONValue.double(2.5).doubleValue == 2.5)
        #expect(AnyLanguageModel.JSONValue.int(42).doubleValue == 42.0)
        #expect(AnyLanguageModel.JSONValue.string("2.5").doubleValue == nil)
        #expect(AnyLanguageModel.JSONValue.null.doubleValue == nil)
    }

    @Test func stringValue() {
        #expect(AnyLanguageModel.JSONValue.string("hello").stringValue == "hello")
        #expect(AnyLanguageModel.JSONValue.int(1).stringValue == nil)
        #expect(AnyLanguageModel.JSONValue.null.stringValue == nil)
    }

    @Test func arrayValue() {
        #expect(sample.objectValue?["array"]?.arrayValue == [.int(1), .string("two"), .null])
        #expect(AnyLanguageModel.JSONValue.array([]).arrayValue == [])
        #expect(sample.arrayValue == nil)
    }

    @Test func objectValue() {
        #expect(sample.objectValue?["int"] == .int(42))
        #expect(AnyLanguageModel.JSONValue.object([:]).objectValue == [:])
        #expect(AnyLanguageModel.JSONValue.string("x").objectValue == nil)
    }

    @Test func accessorsMatchJSONSchema() {
        let values: [AnyLanguageModel.JSONValue] = [
            .null, .bool(true), .int(-3), .double(0.5), .string("s"), .array([.null]), .object(["k": .int(1)]),
        ]
        for value in values {
            let theirs = value.jsonSchemaValue
            #expect(value.isNull == theirs.isNull)
            #expect(value.boolValue == theirs.boolValue)
            #expect(value.intValue == theirs.intValue)
            #expect(value.doubleValue == theirs.doubleValue)
            #expect(value.stringValue == theirs.stringValue)
            #expect(value.arrayValue?.map(\.jsonSchemaValue) == theirs.arrayValue)
            #expect(value.objectValue?.mapValues(\.jsonSchemaValue) == theirs.objectValue)
        }
    }

    // MARK: - JSONSchema conversion

    @Test func convertsToJSONSchemaValue() {
        let converted = sample.jsonSchemaValue
        let expected: JSONSchema.JSONValue = .object([
            "null": .null,
            "bool": .bool(true),
            "int": .int(42),
            "double": .double(2.5),
            "string": .string("hello"),
            "array": .array([.int(1), .string("two"), .null]),
            "object": .object(["nested": .bool(false)]),
        ])
        #expect(converted == expected)
    }

    @Test func convertsFromJSONSchemaValue() {
        let value: JSONSchema.JSONValue = .object([
            "null": .null,
            "bool": .bool(true),
            "int": .int(42),
            "double": .double(2.5),
            "string": .string("hello"),
            "array": .array([.int(1), .string("two"), .null]),
            "object": .object(["nested": .bool(false)]),
        ])
        #expect(AnyLanguageModel.JSONValue(value) == sample)
    }

    @Test func conversionRoundTrips() {
        #expect(AnyLanguageModel.JSONValue(sample.jsonSchemaValue) == sample)
    }
}
