import Foundation
import Testing

@testable import AnyLanguageModel

@Generable
private struct StringifiedToolArguments {
    var count: Int?
    var enabled: Bool?
    var ratio: Double?
}

@Suite("Primitive string coercion")
struct PrimitiveStringCoercionTests {
    @Test func decodesStringifiedNumbersAndBools() throws {
        #expect(try Int(GeneratedContent(kind: .string("5"))) == 5)
        #expect(try Int(GeneratedContent(kind: .string(" -12 "))) == -12)
        #expect(try Int(GeneratedContent(kind: .string("5.0"))) == 5)
        #expect(try Bool(GeneratedContent(kind: .string("true"))) == true)
        #expect(try Bool(GeneratedContent(kind: .string("False"))) == false)
        #expect(try Double(GeneratedContent(kind: .string("3.25"))) == 3.25)
        #expect(try Float(GeneratedContent(kind: .string("0.5"))) == 0.5)
        #expect(try Decimal(GeneratedContent(kind: .string("2.5"))) == Decimal(2.5))
    }

    @Test func rejectsUnparseableStrings() {
        #expect(throws: GeneratedContentConversionError.self) {
            try Int(GeneratedContent(kind: .string("five")))
        }
        #expect(throws: GeneratedContentConversionError.self) {
            try Int(GeneratedContent(kind: .string("5.5")))
        }
        #expect(throws: GeneratedContentConversionError.self) {
            try Bool(GeneratedContent(kind: .string("yes")))
        }
        #expect(throws: GeneratedContentConversionError.self) {
            try Double(GeneratedContent(kind: .string("")))
        }
    }

    @Test func nativeKindsStillDecode() throws {
        #expect(try Int(GeneratedContent(kind: .number(7))) == 7)
        #expect(try Bool(GeneratedContent(kind: .bool(true))) == true)
        #expect(try Double(GeneratedContent(kind: .number(1.5))) == 1.5)
        #expect(try String(GeneratedContent(kind: .string("text"))) == "text")
    }

    @Test func decodesGenerableArgumentsWithStringifiedValues() throws {
        let json = #"{"count": "3", "enabled": "true", "ratio": "0.75"}"#
        let arguments = try StringifiedToolArguments(GeneratedContent(json: json))

        #expect(arguments.count == 3)
        #expect(arguments.enabled == true)
        #expect(arguments.ratio == 0.75)
    }
}
