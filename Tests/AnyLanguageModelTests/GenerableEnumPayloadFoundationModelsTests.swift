import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS)
    import FoundationModels

    private let isFoundationModelsAvailable: Bool = {
        if #available(macOS 26.0, iOS 26.0, watchOS 27.0, visionOS 26.0, *) {
            return true
        }
        return false
    }()

    /// Compares enums with associated values in AnyLanguageModel
    /// with the same enums in Foundation Models,
    /// whose `@Generable` macro comes from the SDK that builds the tests.
    @Suite("Generable enums with associated values in Foundation Models", .enabled(if: isFoundationModelsAvailable))
    struct GenerableEnumPayloadFoundationModelsTests {
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, visionOS 26.0, *)
        private typealias Native = FoundationModelsEnums

        @available(macOS 26.0, iOS 26.0, watchOS 27.0, visionOS 26.0, *)
        private var steps: [(Step, Native.Step)] {
            [
                (.set(field: "count", to: .number(1)), .set(field: "count", to: .number(1))),
                (.add(.name("amount"), to: "count"), .add(.name("amount"), to: "count")),
                (.pair(1, 2), .pair(1, 2)),
                (.note("hi"), .note("hi")),
                (.note(nil), .note(nil)),
                (.reset, .reset),
            ]
        }

        @available(macOS 26.0, iOS 26.0, watchOS 27.0, visionOS 26.0, *)
        private var annotations: [(Annotation, Native.Annotation)] {
            [
                (.alias(nil), .alias(nil)),
                (.generic(nil), .generic(nil)),
                (.qualified(nil), .qualified(nil)),
                (.alias("hi"), .alias("hi")),
                (.generic(3), .generic(3)),
            ]
        }

        /// Foundation Models writes a space after each colon and comma.
        private func compacted(_ json: String) -> String {
            json.replacingOccurrences(of: ": ", with: ":").replacingOccurrences(of: ", ", with: ",")
        }

        @available(macOS 26.0, iOS 26.0, watchOS 27.0, visionOS 26.0, *)
        @Test func contentMatchesFoundationModels() {
            for (step, native) in steps {
                #expect(step.generatedContent.jsonString == compacted(native.generatedContent.jsonString))
            }
            for (annotation, native) in annotations {
                #expect(annotation.generatedContent.jsonString == compacted(native.generatedContent.jsonString))
            }
        }

        @available(macOS 26.0, iOS 26.0, watchOS 27.0, visionOS 26.0, *)
        @Test func foundationModelsContentDecodes() throws {
            for (step, native) in steps {
                let content = try AnyLanguageModel.GeneratedContent(json: native.generatedContent.jsonString)
                #expect(try Step(content) == step)
            }
            for (annotation, native) in annotations {
                let content = try AnyLanguageModel.GeneratedContent(json: native.generatedContent.jsonString)
                #expect(try Annotation(content) == annotation)
            }
        }
    }
#endif
