import Testing

@testable import AnyLanguageModel

/// Constrained generation of `anyOf` schemas: the model, not declaration order, picks the variant.
struct AnyOfGenerationTests {
    private let quote = 0
    private let comma = 1
    private let rightBrace = 2
    private let colon = 4
    private let a = 8
    private let b = 9
    private let x = 10
    private let y = 11
    private let leftBrace = 15
    private let null = 70
    private let one = 6
    private let eos = 50

    private func generate(_ schema: GenerationSchema, sampling queue: [Int]) async throws -> String {
        let tokenToText: [Int: String] = [
            quote: "\"", comma: ",", rightBrace: "}", 3: "]", colon: ":",
            a: "a", b: "b", x: "x", y: "y", leftBrace: "{", null: "null", one: "1", eos: "<eos>",
        ]
        let textToTokens: [String: [Int]] = [
            "\"": [quote], ",": [comma], "}": [rightBrace], ":": [colon], "{": [leftBrace],
            "a": [a], "b": [b], "x": [x], "y": [y], "1": [one], "null": [null],
            "\"x\":": [quote, x, quote, colon],
            "\"y\":": [quote, y, quote, colon],
            ",\"x\":": [comma, quote, x, quote, colon],
            ",\"y\":": [comma, quote, y, quote, colon],
        ]
        let backend = MockTokenBackend(
            tokenToText: tokenToText,
            textToTokens: textToTokens,
            eosToken: eos,
            endTokens: [eos],
            maximumTokens: 64,
            samplingQueue: queue
        )
        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        return try await generator.generate()
    }

    private func letter(_ choices: [String]) -> GenerationSchema.Node {
        .string(.init(enumChoices: choices))
    }

    private func object(_ properties: [String: GenerationSchema.Node]) -> GenerationSchema.Node {
        .object(.init(description: nil, properties: properties, required: Set(properties.keys)))
    }

    /// The shape of an enum with payloads: every case has the key `x`, whose value names the case.
    private func discriminatedUnion() -> GenerationSchema {
        GenerationSchema.primitive(
            String.self,
            node: .anyOf([
                object(["x": letter(["a"])]),
                object(["x": letter(["b"]), "y": letter(["a"])]),
            ])
        )
    }

    @Test func nullableStringCanBeNull() async throws {
        let schema = GenerationSchema.primitive(String.self, node: .anyOf([letter(["a"]), .null]))
        #expect(try await generate(schema, sampling: [null]) == "null")
    }

    @Test func nullFirstUnionCanBeAString() async throws {
        let schema = GenerationSchema.primitive(String.self, node: .anyOf([.null, letter(["a"])]))
        #expect(try await generate(schema, sampling: [quote, a]) == "\"a\"")
    }

    @Test func nullableObjectFromDynamicSchemaCanBeEitherBranch() async throws {
        let person = DynamicGenerationSchema(
            name: "Person",
            properties: [.init(name: "x", schema: DynamicGenerationSchema(name: "Letter", anyOf: ["a"]))]
        )
        let schema = try GenerationSchema(
            root: DynamicGenerationSchema(name: "NullablePerson", anyOf: [person, .null]),
            dependencies: []
        )
        #expect(try await generate(schema, sampling: [null]) == "null")
        #expect(try await generate(schema, sampling: [leftBrace, quote, x, quote, colon, a]) == #"{"x":"a"}"#)
    }

    @Test func objectUnionFollowsTheFirstKey() async throws {
        let schema = GenerationSchema.primitive(
            String.self,
            node: .anyOf([object(["x": letter(["a"])]), object(["y": letter(["a"])])])
        )
        #expect(try await generate(schema, sampling: [quote, y, quote, colon, a]) == #"{"y":"a"}"#)
    }

    @Test func discriminatedUnionFollowsTheSharedKeysValue() async throws {
        let queue = [quote, x, quote, colon, b, comma, quote, y, quote, colon, a]
        #expect(try await generate(discriminatedUnion(), sampling: queue) == #"{"x":"b","y":"a"}"#)
    }

    @Test func discriminatedUnionAcceptsThePayloadKeyFirst() async throws {
        // Only the second case declares `y`, so emitting it first already picks that case.
        let queue = [quote, y, quote, colon, a, comma, quote, x, quote, colon, b]
        #expect(try await generate(discriminatedUnion(), sampling: queue) == #"{"y":"a","x":"b"}"#)
    }

    @Test func sharedKeysThatDifferOnlyInDescriptionKeepBothVariants() async throws {
        // Both variants declare `x` as the same integer except for its description, so
        // emitting `x` must not decide the union; the later `y` picks the second variant.
        let schema = GenerationSchema.primitive(
            String.self,
            node: .anyOf([
                object(["x": .number(.init(description: "first", integerOnly: true))]),
                object([
                    "x": .number(.init(description: "second", integerOnly: true)),
                    "y": letter(["a"]),
                ]),
            ])
        )
        let queue = [quote, x, quote, colon, one, comma, comma, quote, y, quote, colon, a]
        #expect(try await generate(schema, sampling: queue) == #"{"x":1,"y":"a"}"#)
    }

    @Test func discriminatedUnionCanPickThePayloadlessCase() async throws {
        #expect(try await generate(discriminatedUnion(), sampling: [quote, x, quote, colon, a]) == #"{"x":"a"}"#)
    }

    @Test func constantGuidedDiscriminatorPicksTheCase() async throws {
        // Since #304 a `.constant(_:)` guide is a one-choice string enum, so it narrows like any other choice.
        let first = DynamicGenerationSchema(
            name: "First",
            properties: [.init(name: "x", schema: .init(type: String.self, guides: [.constant("a")]))]
        )
        let second = DynamicGenerationSchema(
            name: "Second",
            properties: [
                .init(name: "x", schema: .init(type: String.self, guides: [.constant("b")])),
                .init(name: "y", schema: .init(type: String.self, guides: [.constant("a")])),
            ]
        )
        let schema = try GenerationSchema(
            root: DynamicGenerationSchema(name: "Union", anyOf: [first, second]),
            dependencies: []
        )
        let queue = [quote, x, quote, colon, b, comma, quote, y, quote, colon, a]
        #expect(try await generate(schema, sampling: queue) == #"{"x":"b","y":"a"}"#)
    }

    @Test func variantsThatStartAlikeFallBackToTheFirst() async throws {
        // Documents the known limitation: two strings both start with `"`, so the first is generated.
        let schema = GenerationSchema.primitive(String.self, node: .anyOf([letter(["a"]), letter(["b"])]))
        #expect(try await generate(schema, sampling: [a]) == "\"a\"")
    }
}
