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
}
