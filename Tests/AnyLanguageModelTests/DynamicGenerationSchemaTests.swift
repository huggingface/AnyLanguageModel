import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("DynamicGenerationSchema")
struct DynamicGenerationSchemaTests {
    @Test func objectSchemaConvertsToGenerationSchema() throws {
        let person = DynamicGenerationSchema(
            name: "Person",
            description: "A person object",
            properties: [
                .init(name: "name", description: "Full name", schema: .init(type: String.self)),
                .init(name: "age", schema: .init(type: Int.self), isOptional: true),
            ]
        )

        let schema = try GenerationSchema(root: person, dependencies: [])

        #expect(schema.root == .ref("Person"))
        #expect(schema.defs["Person"] != nil)
    }

    @Test func anyOfSchemaAndStringEnumSchemaConvert() throws {
        let text = DynamicGenerationSchema(type: String.self)
        let integer = DynamicGenerationSchema(type: Int.self)
        let payload = DynamicGenerationSchema(name: "Payload", anyOf: [text, integer])
        let color = DynamicGenerationSchema(name: "Color", anyOf: ["red", "green", "blue"])

        let payloadSchema = try GenerationSchema(root: payload, dependencies: [])
        let colorSchema = try GenerationSchema(root: color, dependencies: [])

        #expect(payloadSchema.root == .ref("Payload"))
        #expect(colorSchema.root == .ref("Color"))
        #expect(payloadSchema.debugDescription.contains("anyOf"))
        #expect(colorSchema.debugDescription.contains("string(enum"))
    }

    @Test func arraySchemaConvertsWithMinAndMax() throws {
        let tags = DynamicGenerationSchema(
            arrayOf: .init(type: String.self),
            minimumElements: 1,
            maximumElements: 3
        )
        let container = DynamicGenerationSchema(
            name: "Container",
            properties: [.init(name: "tags", schema: tags)]
        )

        let schema = try GenerationSchema(root: container, dependencies: [])
        guard case .object(let objectNode) = schema.defs["Container"] else {
            Issue.record("Expected Container definition to be an object")
            return
        }
        guard case .array(let arrayNode) = objectNode.properties["tags"] else {
            Issue.record("Expected tags property to be an array")
            return
        }
        #expect(arrayNode.minItems == 1)
        #expect(arrayNode.maxItems == 3)
    }

    @Test func typeInitializerMapsScalarAndReferenceBodies() {
        let boolSchema = DynamicGenerationSchema(type: Bool.self)
        let stringSchema = DynamicGenerationSchema(type: String.self)
        let intSchema = DynamicGenerationSchema(type: Int.self)
        let floatSchema = DynamicGenerationSchema(type: Float.self)
        let doubleSchema = DynamicGenerationSchema(type: Double.self)
        let decimalSchema = DynamicGenerationSchema(type: Decimal.self)
        let referenceSchema = DynamicGenerationSchema(type: GeneratedContent.self)

        if case .scalar(.bool) = boolSchema.body {} else { Issue.record("Expected bool scalar mapping") }
        if case .scalar(.string) = stringSchema.body {} else { Issue.record("Expected string scalar mapping") }
        if case .scalar(.integer) = intSchema.body {} else { Issue.record("Expected integer scalar mapping") }
        if case .scalar(.number) = floatSchema.body {} else { Issue.record("Expected float number mapping") }
        if case .scalar(.number) = doubleSchema.body {} else { Issue.record("Expected double number mapping") }
        if case .scalar(.decimal) = decimalSchema.body {} else { Issue.record("Expected decimal mapping") }

        if case .reference(let name) = referenceSchema.body {
            #expect(name.contains("GeneratedContent"))
        } else {
            Issue.record("Expected reference mapping for non-scalar Generable type")
        }
    }

    @Test func referenceInitializerCreatesReferenceBody() {
        let reference = DynamicGenerationSchema(referenceTo: "Address")
        if case .reference(let name) = reference.body {
            #expect(name == "Address")
        } else {
            Issue.record("Expected reference body")
        }
    }

    @Test func nullSchemaEncodesAsNullType() throws {
        let person = DynamicGenerationSchema(
            name: "Person",
            properties: [.init(name: "fullName", schema: .init(type: String.self))]
        )
        let nullablePerson = DynamicGenerationSchema(name: "NullablePerson", anyOf: [person, .null])

        let schema = try GenerationSchema(root: nullablePerson, dependencies: [])
        let data = try JSONEncoder().encode(schema)
        let json = try #require(String(data: data, encoding: .utf8))
        #expect(json.contains(#"{"type":"null"}"#))

        let decoded = try JSONDecoder().decode(GenerationSchema.self, from: data)
        #expect(decoded == schema)
    }

    @Test func duplicateDependencyNamesThrow() {
        let dep1 = DynamicGenerationSchema(name: "Shared", properties: [])
        let dep2 = DynamicGenerationSchema(name: "Shared", properties: [])
        let root = DynamicGenerationSchema(referenceTo: "Shared")

        #expect(throws: GenerationSchema.SchemaError.self) {
            _ = try GenerationSchema(root: root, dependencies: [dep1, dep2])
        }
    }

    @Test func undefinedReferenceThrows() {
        let root = DynamicGenerationSchema(referenceTo: "MissingType")

        #expect(throws: GenerationSchema.SchemaError.self) {
            _ = try GenerationSchema(root: root, dependencies: [])
        }
    }

    @Test func encodingIsStableAcrossPropertyAndDefinitionInsertionOrders() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        for _ in 0 ..< 16 {
            let schemas = try Self.schemasInDifferentInsertionOrders()
            #expect(try encoder.encode(schemas[0]) == encoder.encode(schemas[1]))
            for schema in schemas {
                let recorder = SchemaKeyOrderEncoder()
                try schema.encode(to: recorder)
                #expect(recorder.keys["$defs"] == ["Alpha", "Beta", "Root"])
                #expect(recorder.keys["$defs.Root.properties"] == ["alpha", "beta"])
                #expect(recorder.keys["$defs.Alpha.properties"] == ["count", "name"])
                #expect(recorder.keys["$defs.Beta.properties"] == ["enabled", "score"])
            }
        }
    }

    static func schemasInDifferentInsertionOrders() throws -> [GenerationSchema] {
        let alpha = DynamicGenerationSchema(
            name: "Alpha",
            properties: [
                .init(name: "name", schema: .init(type: String.self)),
                .init(name: "count", schema: .init(type: Int.self)),
            ]
        )
        let alphaReversed = DynamicGenerationSchema(
            name: "Alpha",
            properties: [
                .init(name: "count", schema: .init(type: Int.self)),
                .init(name: "name", schema: .init(type: String.self)),
            ]
        )
        let beta = DynamicGenerationSchema(
            name: "Beta",
            properties: [
                .init(name: "enabled", schema: .init(type: Bool.self)),
                .init(name: "score", schema: .init(type: Double.self)),
            ]
        )
        let betaReversed = DynamicGenerationSchema(
            name: "Beta",
            properties: [
                .init(name: "score", schema: .init(type: Double.self)),
                .init(name: "enabled", schema: .init(type: Bool.self)),
            ]
        )
        let alphaFirstRoot = DynamicGenerationSchema(
            name: "Root",
            properties: [
                .init(name: "alpha", schema: .init(referenceTo: "Alpha")),
                .init(name: "beta", schema: .init(referenceTo: "Beta")),
            ]
        )
        let betaFirstRoot = DynamicGenerationSchema(
            name: "Root",
            properties: [
                .init(name: "beta", schema: .init(referenceTo: "Beta")),
                .init(name: "alpha", schema: .init(referenceTo: "Alpha")),
            ]
        )

        let alphaFirst = try GenerationSchema(root: alphaFirstRoot, dependencies: [alpha, beta])
        let betaFirst = try GenerationSchema(root: betaFirstRoot, dependencies: [betaReversed, alphaReversed])

        return [alphaFirst, betaFirst]
    }
}

/// Records schema key visitation before a concrete encoder can rearrange the keys.
private final class SchemaKeyOrderEncoder: Encoder {
    var codingPath: [any CodingKey] = []
    var userInfo: [CodingUserInfoKey: Any] = [:]
    var keys: [String: [String]] = [:]

    func container<Key: CodingKey>(keyedBy type: Key.Type) -> KeyedEncodingContainer<Key> {
        KeyedEncodingContainer(Container<Key>(encoder: self, codingPath: codingPath))
    }

    func unkeyedContainer() -> any UnkeyedEncodingContainer {
        fatalError("This fixture has no array nodes")
    }

    func singleValueContainer() -> any SingleValueEncodingContainer {
        fatalError("Schema nodes use keyed containers")
    }

    private struct Container<Key: CodingKey>: KeyedEncodingContainerProtocol {
        let encoder: SchemaKeyOrderEncoder
        var codingPath: [any CodingKey]

        func record(_ key: Key) {
            let path = codingPath.map(\.stringValue).joined(separator: ".")
            encoder.keys[path, default: []].append(key.stringValue)
        }

        mutating func encode<T: Encodable>(_ value: T, forKey key: Key) throws {
            record(key)
            if let node = value as? GenerationSchema.Node {
                let previousPath = encoder.codingPath
                encoder.codingPath = codingPath + [key]
                defer { encoder.codingPath = previousPath }
                try node.encode(to: encoder)
            }
        }

        mutating func encodeNil(forKey key: Key) throws { record(key) }

        mutating func nestedContainer<NestedKey: CodingKey>(
            keyedBy type: NestedKey.Type,
            forKey key: Key
        ) -> KeyedEncodingContainer<NestedKey> {
            record(key)
            return KeyedEncodingContainer(Container<NestedKey>(encoder: encoder, codingPath: codingPath + [key]))
        }

        mutating func nestedUnkeyedContainer(forKey key: Key) -> any UnkeyedEncodingContainer {
            fatalError("This fixture has no nested unkeyed containers")
        }

        mutating func superEncoder() -> any Encoder { encoder }
        mutating func superEncoder(forKey key: Key) -> any Encoder { encoder }
    }
}
