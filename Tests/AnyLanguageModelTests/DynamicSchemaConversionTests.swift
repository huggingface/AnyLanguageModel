import Testing

@testable import AnyLanguageModel

#if canImport(FoundationModels)
    import FoundationModels
    import JSONSchema

    private let isFoundationModelsAvailable: Bool = {
        if #available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            return true
        }
        return false
    }()

    @Suite("Dynamic Schema Conversion", .enabled(if: isFoundationModelsAvailable))
    struct DynamicSchemaConversionTests {

        // MARK: - Primitive Types

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertStringSchema() throws {
            let schema: JSONSchema = .string(description: "A name")
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertStringSchemaWithEnum() throws {
            let schema: JSONSchema = .string(enum: ["red", "green", "blue"])
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertStringSchemaWithConst() throws {
            let schema: JSONSchema = .string(const: "fixed_value")
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertStringSchemaWithPattern() throws {
            let schema: JSONSchema = .string(pattern: "^[A-Z]{2}$")
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertIntegerSchema() throws {
            let schema: JSONSchema = .integer(description: "An age")
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertIntegerSchemaWithRange() throws {
            let schema: JSONSchema = .integer(minimum: 0, maximum: 100)
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertIntegerSchemaWithEnum() throws {
            let schema: JSONSchema = .integer(enum: [1, 2, 3, 5, 8])
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertNumberSchema() throws {
            let schema: JSONSchema = .number(description: "A temperature")
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertNumberSchemaWithRange() throws {
            let schema: JSONSchema = .number(minimum: -273.15, maximum: 1000.0)
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertBooleanSchema() throws {
            let schema: JSONSchema = .boolean(description: "Is active")
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        // MARK: - Array Types

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertArraySchema() throws {
            let schema: JSONSchema = .array(items: .string())
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertArraySchemaWithConstraints() throws {
            let schema: JSONSchema = .array(items: .integer(), minItems: 1, maxItems: 10)
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertArraySchemaWithoutItems() throws {
            let schema: JSONSchema = .array()
            let dynamic = convertToDynamicSchema(schema)

            // Should default to String items
            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        // MARK: - Object Types

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertObjectSchema() throws {
            let schema: JSONSchema = .object(
                description: "A person",
                properties: [
                    "name": .string(description: "The person's name"),
                    "age": .integer(description: "The person's age"),
                ],
                required: ["name"]
            )
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertNestedObjectSchema() throws {
            let addressSchema: JSONSchema = .object(
                properties: [
                    "street": .string(),
                    "city": .string(),
                    "region": .string(),
                    "postalCode": .string(),
                ],
                required: ["street", "city", "region"]
            )

            let schema: JSONSchema = .object(
                properties: [
                    "name": .string(),
                    "address": addressSchema,
                ],
                required: ["name", "address"]
            )
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertObjectWithArrayProperty() throws {
            let schema: JSONSchema = .object(
                properties: [
                    "tags": .array(items: .string(), minItems: 1),
                    "scores": .array(items: .integer()),
                ],
                required: ["tags"]
            )
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        // MARK: - Composite Types

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertAnyOfSchema() throws {
            let schema: JSONSchema = .anyOf([
                .string(),
                .integer(),
            ])
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertReferenceSchema() {
            let schema: JSONSchema = .reference("SomeType")
            // Reference schemas need the referenced type in dependencies
            // This test just verifies the conversion doesn't crash
            _ = convertToDynamicSchema(schema)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertNamedStringEnumDependencySchema() throws {
            let outputVersion: JSONSchema = .string(enum: ["v0", "v1"])
            let enumDependency = convertToDynamicSchema(outputVersion, name: "OutputVersion")

            let root = FoundationModels.DynamicGenerationSchema(
                name: "Root",
                properties: [
                    .init(
                        name: "version",
                        schema: .init(referenceTo: "OutputVersion")
                    )
                ]
            )

            _ = try FoundationModels.GenerationSchema(root: root, dependencies: [enumDependency])
        }

        // MARK: - Fallback Types

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertNullSchemaFallsBackToString() throws {
            let schema: JSONSchema = .null
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertEmptySchemaFallsBackToString() throws {
            let schema: JSONSchema = .empty
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertAnySchemaFallsBackToString() throws {
            let schema: JSONSchema = .any
            let dynamic = convertToDynamicSchema(schema)

            _ = try FoundationModels.GenerationSchema(root: dynamic, dependencies: [])
        }

        // MARK: - Property Conversion

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertRequiredProperty() throws {
            let schema: JSONSchema = .string(description: "Required field")
            let property = convertToProperty(key: "name", schema: schema, required: ["name"])

            // Build a schema with this property to verify it's valid
            let objectSchema = FoundationModels.DynamicGenerationSchema(
                name: "Test",
                description: nil,
                properties: [property]
            )
            _ = try FoundationModels.GenerationSchema(root: objectSchema, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertOptionalProperty() throws {
            let schema: JSONSchema = .string(description: "Optional field")
            let property = convertToProperty(key: "nickname", schema: schema, required: ["name"])

            let objectSchema = FoundationModels.DynamicGenerationSchema(
                name: "Test",
                description: nil,
                properties: [property]
            )
            _ = try FoundationModels.GenerationSchema(root: objectSchema, dependencies: [])
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertPropertyWithDescription() throws {
            let schema: JSONSchema = .string(description: "A detailed description")
            let property = convertToProperty(key: "field", schema: schema, required: [])

            let objectSchema = FoundationModels.DynamicGenerationSchema(
                name: "Test",
                description: nil,
                properties: [property]
            )
            _ = try FoundationModels.GenerationSchema(root: objectSchema, dependencies: [])
        }

        // MARK: - Constant Value Conversion

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertIntConstant() {
            let value: JSONValue = .int(42)
            let schema = convertConstToSchema(value)

            #expect(schema != nil)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertDoubleConstant() {
            let value: JSONValue = .double(3.14)
            let schema = convertConstToSchema(value)

            #expect(schema != nil)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertStringConstant() {
            let value: JSONValue = .string("constant")
            let schema = convertConstToSchema(value)

            #expect(schema != nil)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertNullConstantReturnsNil() {
            let value: JSONValue = .null
            let schema = convertConstToSchema(value)

            #expect(schema == nil)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertBoolConstantReturnsNil() {
            let value: JSONValue = .bool(true)
            let schema = convertConstToSchema(value)

            #expect(schema == nil)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertArrayConstantReturnsNil() {
            let value: JSONValue = .array([.int(1), .int(2)])
            let schema = convertConstToSchema(value)

            #expect(schema == nil)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertObjectConstantReturnsNil() {
            let value: JSONValue = .object(["key": .string("value")])
            let schema = convertConstToSchema(value)

            #expect(schema == nil)
        }

        // MARK: - Integration with AnyLanguageModel.GenerationSchema

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertFromAnyLanguageModelGenerationSchema() {
            // Create a schema using AnyLanguageModel types
            let schema = AnyLanguageModel.GenerationSchema(
                type: String.self,
                properties: [
                    AnyLanguageModel.GenerationSchema.Property(
                        name: "text",
                        description: "Some text",
                        type: String.self
                    )
                ]
            )

            // Convert through the FoundationModels.GenerationSchema initializer
            _ = FoundationModels.GenerationSchema(schema)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertSchemaWithIntegerProperty() {
            let schema = AnyLanguageModel.GenerationSchema(
                type: Int.self,
                properties: [
                    AnyLanguageModel.GenerationSchema.Property(
                        name: "count",
                        description: "A count value",
                        type: Int.self
                    )
                ]
            )

            _ = FoundationModels.GenerationSchema(schema)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertSchemaWithBooleanProperty() {
            let schema = AnyLanguageModel.GenerationSchema(
                type: Bool.self,
                properties: [
                    AnyLanguageModel.GenerationSchema.Property(
                        name: "isEnabled",
                        description: "To enable or not to enable",
                        type: Bool.self
                    )
                ]
            )

            _ = FoundationModels.GenerationSchema(schema)
        }

        @available(macOS 26.0, iOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
        @Test func convertSchemaWithMultiplePropertyTypes() {
            let schema = AnyLanguageModel.GenerationSchema(
                type: String.self,
                properties: [
                    AnyLanguageModel.GenerationSchema.Property(
                        name: "name",
                        description: "A name",
                        type: String.self
                    ),
                    AnyLanguageModel.GenerationSchema.Property(
                        name: "age",
                        description: "An age",
                        type: Int.self
                    ),
                    AnyLanguageModel.GenerationSchema.Property(
                        name: "active",
                        description: "Is active",
                        type: Bool.self
                    ),
                ]
            )

            _ = FoundationModels.GenerationSchema(schema)
        }
    }
#endif
