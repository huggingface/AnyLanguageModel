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

        /// The encoded definitions of each case in a schema, by the name in its `type`.
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, visionOS 26.0, *)
        private func caseDefinitions(_ schema: FoundationModels.GenerationSchema) throws -> [String: NSDictionary] {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .sortedKeys
            let object = try JSONSerialization.jsonObject(with: encoder.encode(schema)) as? [String: Any]
            let defs = try #require(object?["$defs"] as? [String: NSDictionary])
            var cases: [String: NSDictionary] = [:]
            for definition in defs.values {
                guard let type = (definition["properties"] as? [String: NSDictionary])?["type"],
                    let name = type["const"] as? String ?? (type["enum"] as? [String])?.first
                else { continue }
                cases[name] = definition
            }
            return cases
        }

        /// Converted for the system model, each case's schema encodes as Foundation Models' own does,
        /// apart from the names of definitions in `$ref`s, the default descriptions,
        /// and the order of properties, which the conversion doesn't keep.
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, visionOS 26.0, *)
        @Test func caseSchemasMatchFoundationModels() throws {
            let converted = try caseDefinitions(FoundationModels.GenerationSchema(Step.generationSchema))
            let native = try caseDefinitions(Native.Step.generationSchema)

            #expect(Set(converted.keys) == Set(native.keys))
            #expect(Set(native.keys).isSuperset(of: ["set", "add", "pair", "note", "reset", "name", "number"]))
            for (name, nativeCase) in native {
                let convertedCase = try #require(converted[name])
                #expect(
                    Set(convertedCase["required"] as? [String] ?? []) == Set(nativeCase["required"] as? [String] ?? [])
                )
                let properties = try #require(convertedCase["properties"] as? [String: NSDictionary])
                let nativeProperties = try #require(nativeCase["properties"] as? [String: NSDictionary])
                #expect(Set(properties.keys) == Set(nativeProperties.keys))
                for (key, nativeProperty) in nativeProperties where nativeProperty["$ref"] == nil {
                    #expect(properties[key] == nativeProperty, "\(name).\(key)")
                }
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
