import Testing

@testable import AnyLanguageModel

@Suite("Strict response schemas")
struct StrictResponseSchemaTests {
    private let address: JSONValue = [
        "type": "object",
        "properties": ["street": ["type": "string"], "unit": ["type": "string"]],
        "required": ["street"],
        "additionalProperties": false,
    ]

    private let strictAddress: JSONValue = [
        "type": "object",
        "properties": ["street": ["type": "string"], "unit": ["type": ["string", "null"]]],
        "required": ["street", "unit"],
        "additionalProperties": false,
    ]

    @Test func nestedObjectRequiresOptionalProperty() {
        let schema: JSONValue = [
            "type": "object",
            "properties": ["name": ["type": "string"], "address": address],
            "required": ["address", "name"],
        ]
        let expected: JSONValue = [
            "type": "object",
            "properties": ["name": ["type": "string"], "address": strictAddress],
            "required": ["address", "name"],
            "additionalProperties": false,
        ]
        #expect(schema.applyingStrictMode() == expected)
    }

    @Test func arrayItemsRequireOptionalProperty() {
        let schema: JSONValue = [
            "type": "object",
            "properties": ["previous": ["type": "array", "items": address]],
            "required": ["previous"],
            "additionalProperties": false,
        ]
        let expected: JSONValue = [
            "type": "object",
            "properties": ["previous": ["type": "array", "items": strictAddress]],
            "required": ["previous"],
            "additionalProperties": false,
        ]
        #expect(schema.applyingStrictMode() == expected)
    }

    @Test func enumCaseRequiresOptionalValue() {
        let note: JSONValue = [
            "type": "object",
            "properties": ["type": ["type": "string", "enum": ["note"]], "value": ["type": "string"]],
            "required": ["type"],
            "additionalProperties": false,
        ]
        let schema: JSONValue = [
            "type": "object",
            "properties": ["response": ["anyOf": [note]]],
            "required": ["response"],
            "additionalProperties": false,
        ]
        let strictNote: JSONValue = [
            "type": "object",
            "properties": [
                "type": ["type": "string", "enum": ["note"]],
                "value": ["type": ["string", "null"]],
            ],
            "required": ["type", "value"],
            "additionalProperties": false,
        ]
        let expected: JSONValue = [
            "type": "object",
            "properties": ["response": ["anyOf": [strictNote]]],
            "required": ["response"],
            "additionalProperties": false,
        ]
        #expect(schema.applyingStrictMode() == expected)
    }

    /// A type with listed values, an object, or a union allows `null` as another choice of a union,
    /// and a schema that already allows `null` is unchanged.
    @Test func optionalPropertiesAllowNull() {
        let choice: JSONValue = ["type": "string", "enum": ["a", "b"]]
        let union: JSONValue = ["anyOf": [["type": "string"], ["type": "integer"]]]
        let nullable: JSONValue = ["type": ["integer", "null"]]
        let schema: JSONValue = [
            "type": "object",
            "properties": ["choice": choice, "address": address, "union": union, "nullable": nullable],
            "required": [],
        ]
        let null: JSONValue = ["type": "null"]
        let expected: JSONValue = [
            "type": "object",
            "properties": [
                "choice": ["anyOf": [choice, null]],
                "address": ["anyOf": [strictAddress, null]],
                "union": ["anyOf": [["type": "string"], ["type": "integer"], null]],
                "nullable": nullable,
            ],
            "required": ["address", "choice", "nullable", "union"],
            "additionalProperties": false,
        ]
        #expect(schema.applyingStrictMode() == expected)
    }
}
