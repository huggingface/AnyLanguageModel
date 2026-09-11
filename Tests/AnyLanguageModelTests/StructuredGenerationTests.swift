import Testing

@testable import AnyLanguageModel

struct StructuredGenerationTests {
    private func baseTokenMaps() -> (tokenToText: [Int: String], textToTokens: [String: [Int]]) {
        let quoteToken = 0
        let commaToken = 1
        let rightBraceToken = 2
        let rightBracketToken = 3
        let colonToken = 4
        let zeroToken = 5
        let oneToken = 6
        let twoToken = 7
        let aToken = 8
        let bToken = 9
        let xToken = 10
        let yToken = 11
        let zToken = 12
        let minusToken = 13
        let minusOneToken = 14
        let eosToken = 50

        let tokenToText: [Int: String] = [
            quoteToken: "\"",
            commaToken: ",",
            rightBraceToken: "}",
            rightBracketToken: "]",
            colonToken: ":",
            zeroToken: "0",
            oneToken: "1",
            twoToken: "2",
            aToken: "a",
            bToken: "b",
            xToken: "x",
            yToken: "y",
            zToken: "z",
            minusToken: "-",
            minusOneToken: "-1",
            eosToken: "<eos>",
        ]

        let textToTokens: [String: [Int]] = [
            "\"": [quoteToken],
            ",": [commaToken],
            "}": [rightBraceToken],
            "]": [rightBracketToken],
            ":": [colonToken],
            "0": [zeroToken],
            "1": [oneToken],
            "2": [twoToken],
            "a": [aToken],
            "b": [bToken],
            "x": [xToken],
            "y": [yToken],
            "z": [zToken],
            "-": [minusToken],
            "-1": [minusOneToken],
            "ab": [aToken, bToken],
        ]

        return (tokenToText, textToTokens)
    }

    @Test func numberOutOfRangeThrows() async throws {
        let maps = baseTokenMaps()
        let numberNode = GenerationSchema.NumberNode(
            description: nil,
            minimum: 0,
            maximum: 10,
            integerOnly: true
        )
        let schema = GenerationSchema.primitive(Int.self, node: .number(numberNode))
        let eosToken = 50
        let rightBraceToken = 2
        let oneToken = 6
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 8,
            samplingQueue: [oneToken, oneToken, rightBraceToken]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        do {
            _ = try await generator.generate()
            Issue.record("Expected number out-of-range error.")
        } catch let error as ConstrainedGenerationError {
            guard case .numberOutOfRange = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func patternMismatchThrows() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: "^abc$",
            enumChoices: nil
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let quoteToken = 0
        let xToken = 10
        let yToken = 11
        let zToken = 12
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 8,
            samplingQueue: [xToken, yToken, zToken, quoteToken]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        do {
            _ = try await generator.generate()
            Issue.record("Expected pattern mismatch error.")
        } catch let error as ConstrainedGenerationError {
            guard case .patternMismatch = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func emptyStringEnumProducesEmptyValue() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: ["", "a"]
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 5
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "\"\"")
    }

    @Test func prefixEnumSelectsLongerCandidateDeterministically() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: ["a", "ab"]
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 4
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "\"ab\"")
    }

    @Test func eosStopsGenerationAndReturnsPartialOutput() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: nil
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let aToken = 8
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 4,
            samplingQueue: [aToken, eosToken]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "\"a")
    }

    @Test func multiTokenStructuralEncodingThrows() throws {
        var maps = baseTokenMaps()
        let eosToken = 50
        maps.textToTokens[","] = [1, 1]
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: nil
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))

        do {
            _ = try ConstrainedJSONGenerator(
                backend: MockTokenBackend(
                    tokenToText: maps.tokenToText,
                    textToTokens: maps.textToTokens,
                    eosToken: eosToken,
                    endTokens: [eosToken],
                    maximumTokens: 4
                ),
                schema: schema
            )
            Issue.record("Expected unsupported tokenizer error.")
        } catch let error as ConstrainedGenerationError {
            guard case .unsupportedTokenizer = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func outputMatchesDecodedTokens() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: nil
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let quoteToken = 0
        let aToken = 8
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 4,
            samplingQueue: [aToken, quoteToken]
        )
        let capture = backend.capture

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == capture.decodedText)
    }

    @Test func negativeIntegerWithinRange() async throws {
        let maps = baseTokenMaps()
        let numberNode = GenerationSchema.NumberNode(
            description: nil,
            minimum: -10,
            maximum: 0,
            integerOnly: true
        )
        let schema = GenerationSchema.primitive(Int.self, node: .number(numberNode))
        let eosToken = 50
        let minusOneToken = 14
        let rightBraceToken = 2
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 6,
            samplingQueue: [minusOneToken, rightBraceToken]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "-1")
    }

    @Test func decimalOutOfRangeThrows() async throws {
        let maps = baseTokenMaps()
        let numberNode = GenerationSchema.NumberNode(
            description: nil,
            minimum: nil,
            maximum: 1,
            integerOnly: false
        )
        let schema = GenerationSchema.primitive(Double.self, node: .number(numberNode))
        let eosToken = 50
        let twoToken = 7
        let rightBraceToken = 2
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 6,
            samplingQueue: [twoToken, rightBraceToken]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        do {
            _ = try await generator.generate()
            Issue.record("Expected number out-of-range error.")
        } catch let error as ConstrainedGenerationError {
            guard case .numberOutOfRange = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func tokenBudgetExceededThrows() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: ["a"]
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))
        let eosToken = 50
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 0
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        do {
            _ = try await generator.generate()
            Issue.record("Expected token budget exceeded error.")
        } catch let error as ConstrainedGenerationError {
            guard case .tokenBudgetExceeded = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func anyOfSingleVariantUsesOnlyChoice() async throws {
        let maps = baseTokenMaps()
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: ["a"]
        )
        let schema = GenerationSchema.primitive(
            String.self,
            node: .anyOf([.string(stringNode)])
        )
        let eosToken = 50
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 4
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "\"a\"")
    }

    @Test func multiTokenQuoteEncodingThrows() throws {
        var maps = baseTokenMaps()
        let eosToken = 50
        maps.textToTokens["\""] = [0, 0]
        let stringNode = GenerationSchema.StringNode(
            description: nil,
            pattern: nil,
            enumChoices: nil
        )
        let schema = GenerationSchema.primitive(String.self, node: .string(stringNode))

        do {
            _ = try ConstrainedJSONGenerator(
                backend: MockTokenBackend(
                    tokenToText: maps.tokenToText,
                    textToTokens: maps.textToTokens,
                    eosToken: eosToken,
                    endTokens: [eosToken],
                    maximumTokens: 4
                ),
                schema: schema
            )
            Issue.record("Expected unsupported tokenizer error.")
        } catch let error as ConstrainedGenerationError {
            guard case .unsupportedTokenizer = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    @Test func invalidArrayBoundsThrows() async throws {
        let maps = baseTokenMaps()
        let arrayNode = GenerationSchema.ArrayNode(
            description: nil,
            items: .string(.init()),
            minItems: 3,
            maxItems: 1
        )
        let schema = GenerationSchema.primitive([String].self, node: .array(arrayNode))
        let eosToken = 50
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 6
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        do {
            _ = try await generator.generate()
            Issue.record("Expected invalid array bounds error.")
        } catch let error as ConstrainedGenerationError {
            guard case .invalidArrayBounds = error else {
                Issue.record("Unexpected error: \(error).")
                return
            }
        }
    }

    // MARK: - Model-driven array length

    @Test func arrayLengthIsChosenBySamplingNotBudget() async throws {
        let maps = baseTokenMaps()
        let arrayNode = GenerationSchema.ArrayNode(
            description: nil,
            items: .string(.init(enumChoices: ["a"])),
            minItems: 1,
            maxItems: 3
        )
        let schema = GenerationSchema.primitive([String].self, node: .array(arrayNode))
        let eosToken = 50
        let aToken = 8
        let rightBracket = 3

        // minItems=1 forces first element; model then closes (length 1), not budget-derived 3.
        // With maximumTokens 17 the old formula was minItems + (17 % 3) = 3.
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 17,
            samplingQueue: [
                aToken,  // first "a"
                rightBracket,  // close after 1 (`,` would continue)
            ]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "[\"a\"]")
    }

    @Test func arrayLengthVariesWithSamplingQueue() async throws {
        let maps = baseTokenMaps()
        let arrayNode = GenerationSchema.ArrayNode(
            description: nil,
            items: .string(.init(enumChoices: ["a"])),
            minItems: 1,
            maxItems: 3
        )
        let schema = GenerationSchema.primitive([String].self, node: .array(arrayNode))
        let eosToken = 50
        let aToken = 8
        let comma = 1
        let rightBracket = 3

        // Emit two elements then close — different length than the previous test.
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 64,
            samplingQueue: [
                aToken,  // "a"
                comma,  // continue
                aToken,  // "a"
                rightBracket,  // close
            ]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "[\"a\",\"a\"]")
    }

    @Test func arrayRespectsMaxItems() async throws {
        let maps = baseTokenMaps()
        let arrayNode = GenerationSchema.ArrayNode(
            description: nil,
            items: .string(.init(enumChoices: ["a"])),
            minItems: 1,
            maxItems: 2
        )
        let schema = GenerationSchema.primitive([String].self, node: .array(arrayNode))
        let eosToken = 50
        let aToken = 8
        let comma = 1

        // Model always continues; generator must still stop at maxItems=2.
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 64,
            samplingQueue: [
                aToken,
                comma,
                aToken,
                // further commas would be illegal once max is reached — close is forced
            ]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "[\"a\",\"a\"]")
    }

    @Test func arrayRespectsMinItemsBeforeClose() async throws {
        let maps = baseTokenMaps()
        let arrayNode = GenerationSchema.ArrayNode(
            description: nil,
            items: .string(.init(enumChoices: ["a"])),
            minItems: 2,
            maxItems: 4
        )
        let schema = GenerationSchema.primitive([String].self, node: .array(arrayNode))
        let eosToken = 50
        let aToken = 8
        let rightBracket = 3

        // After first element, `]` is not offered — only forced `,` + second element, then close.
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 64,
            samplingQueue: [
                aToken,  // first
                // no close offered here — comma is emitted forcibly
                aToken,  // second (satisfies minItems)
                rightBracket,  // model closes
            ]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "[\"a\",\"a\"]")
    }

    @Test func emptyArrayWhenModelClosesImmediately() async throws {
        let maps = baseTokenMaps()
        let arrayNode = GenerationSchema.ArrayNode(
            description: nil,
            items: .string(.init(enumChoices: ["a"])),
            minItems: nil,
            maxItems: nil
        )
        let schema = GenerationSchema.primitive([String].self, node: .array(arrayNode))
        let eosToken = 50
        let rightBracket = 3

        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 64,
            samplingQueue: [rightBracket]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "[]")
    }

    @Test func emptyArrayProbeAdmitsMergedTokens() async throws {
        var maps = baseTokenMaps()
        // Byte-pair vocabularies carry the item start and the close on merged tokens.
        let spacedBracket = 60
        let quoteA = 61
        maps.tokenToText[spacedBracket] = " ]"
        maps.tokenToText[quoteA] = "\"a"
        let arrayNode = GenerationSchema.ArrayNode(
            description: nil,
            items: .string(.init(enumChoices: ["a"])),
            minItems: nil,
            maxItems: 1
        )
        let schema = GenerationSchema.primitive([String].self, node: .array(arrayNode))
        let eosToken = 50

        // The probe must offer the whitespace-prefixed close, and choosing it closes the array.
        let closing = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 64,
            samplingQueue: [spacedBracket]
        )
        var closingGenerator = try ConstrainedJSONGenerator(backend: closing, schema: schema)
        #expect(try await closingGenerator.generate() == "[]")

        // The probe must offer the merged item start, and choosing it fills the array.
        let filling = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 64,
            samplingQueue: [quoteA]
        )
        var fillingGenerator = try ConstrainedJSONGenerator(backend: filling, schema: schema)
        #expect(try await fillingGenerator.generate() == "[\"a\"]")
    }

    @Test func arrayTruncatesUnderBudgetPressure() async throws {
        let maps = baseTokenMaps()
        let arrayNode = GenerationSchema.ArrayNode(
            description: nil,
            items: .string(.init(enumChoices: ["a"])),
            minItems: 1,
            maxItems: 8
        )
        let schema = GenerationSchema.primitive([String].self, node: .array(arrayNode))
        let eosToken = 50
        let aToken = 8
        let comma = 1

        // Mock maps encode `]` / `"` / `a` but not `[`, so the opening bracket is free.
        // First element costs 3 tokens (`"`, `a`, `"`). Floor is max(8, budget/10)=8.
        // With budget 11, remaining after the first element is 8 → not strictly greater
        // than the floor → force-close. Sampling queue would happily continue with `,`.
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 11,
            samplingQueue: [
                aToken,
                comma,  // would continue if offered — must not be consumed if we truncate
                aToken,
            ]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        // Force-close after satisfying minItems under budget pressure.
        #expect(result == "[\"a\"]")
    }

    // MARK: - Model-driven optional object properties

    private func objectTokenMaps() -> (
        tokenToText: [Int: String],
        textToTokens: [String: [Int]]
    ) {
        // Structural + single-letter keys so `"x":` / `,"y":` tokenize without collisions.
        var maps = baseTokenMaps()
        let quote = 0
        let comma = 1
        let colon = 4
        let x = 10
        let y = 11
        let z = 12
        maps.textToTokens["\"x\":"] = [quote, x, quote, colon]
        maps.textToTokens["\"y\":"] = [quote, y, quote, colon]
        maps.textToTokens["\"z\":"] = [quote, z, quote, colon]
        maps.textToTokens[",\"x\":"] = [comma, quote, x, quote, colon]
        maps.textToTokens[",\"y\":"] = [comma, quote, y, quote, colon]
        maps.textToTokens[",\"z\":"] = [comma, quote, z, quote, colon]
        return maps
    }

    private func allOptionalObjectSchema() -> GenerationSchema {
        let stringNode = GenerationSchema.Node.string(.init(enumChoices: ["a"]))
        let objectNode = GenerationSchema.ObjectNode(
            description: nil,
            properties: [
                "x": stringNode,
                "y": stringNode,
                "z": stringNode,
            ],
            required: []
        )
        // Type argument is unused; only the node shapes generation.
        return GenerationSchema.primitive(String.self, node: .object(objectNode))
    }

    @Test func optionalObjectKeysAreChosenBySamplingNotNameHash() async throws {
        let maps = objectTokenMaps()
        let schema = allOptionalObjectSchema()
        let eosToken = 50
        let quote = 0
        let y = 11
        let aToken = 8
        let colon = 4
        let rightBrace = 2

        // Open with "y": (not lexicographically first), value "a", then close — leave x/z out.
        // generateChoice samples every token of the chosen property-start and the enum value.
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 64,
            samplingQueue: [
                quote, y, quote, colon,  // "y":
                aToken,  // enum value "a"
                rightBrace,  // close
            ]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == #"{"y":"a"}"#)
        #expect(!result.contains("\"x\""))
        #expect(!result.contains("\"z\""))
    }

    @Test func optionalObjectKeysVaryWithSamplingQueue() async throws {
        let maps = objectTokenMaps()
        let schema = allOptionalObjectSchema()
        let eosToken = 50
        let quote = 0
        let x = 10
        let z = 12
        let aToken = 8
        let comma = 1
        let colon = 4
        let rightBrace = 2

        // Emit x then z (skip y) — different key set than the previous test.
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 64,
            samplingQueue: [
                quote, x, quote, colon,  // "x":
                aToken,  // "a"
                comma, quote, z, quote, colon,  // ,"z":
                aToken,  // "a"
                rightBrace,
            ]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == #"{"x":"a","z":"a"}"#)
        #expect(!result.contains("\"y\""))
    }

    @Test func requiredObjectKeysMustBeEmittedBeforeClose() async throws {
        let maps = objectTokenMaps()
        let stringNode = GenerationSchema.Node.string(.init(enumChoices: ["a"]))
        let objectNode = GenerationSchema.ObjectNode(
            description: nil,
            properties: [
                "x": stringNode,
                "y": stringNode,
            ],
            required: ["x"]
        )
        let schema = GenerationSchema.primitive(String.self, node: .object(objectNode))
        let eosToken = 50
        let quote = 0
        let x = 10
        let aToken = 8
        let colon = 4
        let rightBrace = 2

        // "}" is not among candidates until required "x" is emitted.
        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 64,
            samplingQueue: [
                quote, x, quote, colon,
                aToken,
                rightBrace,
            ]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == #"{"x":"a"}"#)
    }

    @Test func emptyObjectWhenModelClosesImmediately() async throws {
        let maps = objectTokenMaps()
        let schema = allOptionalObjectSchema()
        let eosToken = 50
        let rightBrace = 2

        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 64,
            samplingQueue: [rightBrace]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "{}")
    }

    // MARK: - Decimal / number token mask

    private func numberTokenMaps() -> (
        tokenToText: [Int: String],
        textToTokens: [String: [Int]]
    ) {
        var maps = baseTokenMaps()
        let dotToken = 20
        maps.tokenToText[dotToken] = "."
        maps.textToTokens["."] = [dotToken]
        return maps
    }

    @Test func decimalNumberEmitsStandaloneDot() async throws {
        // Qwen2.5-style tokenization of 473.00 is 4 7 3 . 0 0. Standalone `.` must be
        // in the decimal mask; otherwise the model cannot place the point and pads digits
        // until the token cap (re-serialized as e+31 after Double conversion).
        var maps = numberTokenMaps()
        maps.tokenToText[30] = "4"
        maps.tokenToText[31] = "7"
        maps.tokenToText[32] = "3"
        maps.textToTokens["4"] = [30]
        maps.textToTokens["7"] = [31]
        maps.textToTokens["3"] = [32]
        let numberNode = GenerationSchema.NumberNode(
            description: nil,
            minimum: nil,
            maximum: nil,
            integerOnly: false
        )
        let schema = GenerationSchema.primitive(Double.self, node: .number(numberNode))
        let eosToken = 50
        let fourToken = 30
        let sevenToken = 31
        let threeToken = 32
        let dotToken = 20
        let zeroToken = 5
        let rightBrace = 2

        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 64,
            samplingQueue: [
                fourToken, sevenToken, threeToken,  // 473
                dotToken, zeroToken, zeroToken,  // .00
                rightBrace,  // terminate
            ]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "473.00")
    }

    @Test func standaloneMinusIsAllowedInIntegerMask() async throws {
        // Standalone `-` then digit, as BPE tokenizers encode negatives.
        let maps = baseTokenMaps()
        let numberNode = GenerationSchema.NumberNode(
            description: nil,
            minimum: -10,
            maximum: 0,
            integerOnly: true
        )
        let schema = GenerationSchema.primitive(Int.self, node: .number(numberNode))
        let eosToken = 50
        let minusToken = 13
        let oneToken = 6
        let rightBrace = 2

        let backend = MockTokenBackend(
            tokenToText: maps.tokenToText,
            textToTokens: maps.textToTokens,
            eosToken: eosToken,
            endTokens: [eosToken],
            maximumTokens: 8,
            samplingQueue: [minusToken, oneToken, rightBrace]
        )

        var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema)
        let result = try await generator.generate()
        #expect(result == "-1")
    }
}
