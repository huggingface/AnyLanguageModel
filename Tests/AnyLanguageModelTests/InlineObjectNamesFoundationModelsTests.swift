import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(FoundationModels)
    import FoundationModels

    @Suite("Inline objects for Foundation Models")
    struct InlineObjectNamesFoundationModelsTests {
        /// A page, its items and their kinds written in place, with no definitions: the shape a
        /// schema from JSON has.
        static let page = #"""
            {"type":"object","properties":{"page":{"type":"object","properties":{"items":{"type":"array",
              "items":{"type":"object","properties":{
                "kind":{"type":"string","oneOf":[{"const":"text","description":"Words to read."},
                  {"const":"button","description":"A button."}]},
                "words":{"type":"string"}},"required":["kind","words"]}}},"required":["items"]}},
              "required":["page"]}
            """#

        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        struct PageTool: FoundationModels.Tool {
            typealias Arguments = FoundationModels.GeneratedContent
            let name = "show_page"
            let description = "Shows a page."
            let parameters: FoundationModels.GenerationSchema
            func call(arguments: FoundationModels.GeneratedContent) async throws -> String { "shown" }
        }

        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        static func schema() throws -> FoundationModels.GenerationSchema {
            FoundationModels.GenerationSchema(
                try JSONDecoder().decode(AnyLanguageModel.GenerationSchema.self, from: Data(page.utf8)))
        }

        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test func everyObjectAndChoiceIsNamed() throws {
            let json = String(decoding: try JSONEncoder().encode(try Self.schema()), as: UTF8.self)
            #expect(!json.contains(##""$ref":"#""##))
        }

        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test(.enabled(if: SystemLanguageModel.default.isAvailable))
        func aToolTakingInlineObjectsStartsASession() async throws {
            let session = FoundationModels.LanguageModelSession(tools: [PageTool(parameters: try Self.schema())])
            let response = try await session.respond(to: "hello!")
            #expect(!response.content.isEmpty)
        }
    }
#endif
