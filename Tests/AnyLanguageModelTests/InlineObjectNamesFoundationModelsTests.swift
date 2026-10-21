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
                try JSONDecoder().decode(AnyLanguageModel.GenerationSchema.self, from: Data(page.utf8))
            )
        }

        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test func everyObjectAndChoiceIsNamed() throws {
            let json = String(decoding: try JSONEncoder().encode(try Self.schema()), as: UTF8.self)
            #expect(!json.contains(##""$ref":"#""##))
        }

        /// An inline object at the root and one in a definition, under the same property name,
        /// stay two schemas, each with its own properties.
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test func sameInlineNameInRootAndDefinitionStaysApart() throws {
            let json = #"""
                {"type":"object","properties":{
                  "x":{"type":"object","properties":{"fromRoot":{"type":"string"}},"required":["fromRoot"]},
                  "a":{"$ref":"#/$defs/A"}},"required":["x","a"],
                 "$defs":{"A":{"type":"object","properties":{
                  "x":{"type":"object","properties":{"fromA":{"type":"string"}},"required":["fromA"]}},
                  "required":["x"]}}}
                """#
            let schema = FoundationModels.GenerationSchema(
                try JSONDecoder().decode(AnyLanguageModel.GenerationSchema.self, from: Data(json.utf8))
            )
            let encoded = String(decoding: try JSONEncoder().encode(schema), as: UTF8.self)
            #expect(encoded.contains("fromRoot"))
            #expect(encoded.contains("fromA"))
        }

        /// A property named `a.b` and a property `b` of a property `a`, whose paths read the same,
        /// stay two schemas, each with its own properties.
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test func dottedPropertyAndNestedPropertyStayApart() throws {
            let json = #"""
                {"type":"object","properties":{
                  "a.b":{"type":"object","properties":{"fromDotted":{"type":"string"}},"required":["fromDotted"]},
                  "a":{"type":"object","properties":{
                    "b":{"type":"object","properties":{"fromNested":{"type":"string"}},"required":["fromNested"]}},
                    "required":["b"]}},"required":["a.b","a"]}
                """#
            let schema = FoundationModels.GenerationSchema(
                try JSONDecoder().decode(AnyLanguageModel.GenerationSchema.self, from: Data(json.utf8))
            )
            let encoded = String(decoding: try JSONEncoder().encode(schema), as: UTF8.self)
            #expect(encoded.contains("fromDotted"))
            #expect(encoded.contains("fromNested"))
        }

        /// A definition's inline object and a definition named for that object's path stay two
        /// schemas, each with its own properties.
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test func inlineNameMatchingADefinitionStaysApart() throws {
            let json = #"""
                {"type":"object","properties":{
                  "a":{"$ref":"#/$defs/A"},"ax":{"$ref":"#/$defs/A.x"}},"required":["a","ax"],
                 "$defs":{
                  "A":{"type":"object","properties":{
                    "x":{"type":"object","properties":{"fromInline":{"type":"string"}},"required":["fromInline"]}},
                    "required":["x"]},
                  "A.x":{"type":"object","properties":{"fromDefinition":{"type":"string"}},
                    "required":["fromDefinition"]}}}
                """#
            let schema = FoundationModels.GenerationSchema(
                try JSONDecoder().decode(AnyLanguageModel.GenerationSchema.self, from: Data(json.utf8))
            )
            let encoded = String(decoding: try JSONEncoder().encode(schema), as: UTF8.self)
            #expect(encoded.contains("fromInline"))
            #expect(encoded.contains("fromDefinition"))
        }

        /// The items of a definition that is an array, an object and a choice of two objects
        /// written in place, are named from the definition's name and keep their properties.
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test func arrayDefinitionItemsAreNamed() throws {
            let json = #"""
                {"type":"object","properties":{
                  "list":{"$ref":"#/$defs/List"},"mixed":{"$ref":"#/$defs/Mixed"}},"required":["list","mixed"],
                 "$defs":{
                  "List":{"type":"array","items":{"type":"object","properties":{"fromItem":{"type":"string"}},
                    "required":["fromItem"]}},
                  "Mixed":{"type":"array","items":{"anyOf":[
                    {"type":"object","properties":{"fromFirst":{"type":"string"}},"required":["fromFirst"]},
                    {"type":"object","properties":{"fromSecond":{"type":"string"}},"required":["fromSecond"]}]}}}}
                """#
            let schema = FoundationModels.GenerationSchema(
                try JSONDecoder().decode(AnyLanguageModel.GenerationSchema.self, from: Data(json.utf8))
            )
            let encoded = String(decoding: try JSONEncoder().encode(schema), as: UTF8.self)
            #expect(!encoded.contains(##""$ref":"#""##))
            #expect(!encoded.contains(##""title":"""##))
            #expect(encoded.contains("fromItem"))
            #expect(encoded.contains("fromFirst"))
            #expect(encoded.contains("fromSecond"))
        }

        #if !os(watchOS)
            @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
            @Test(.enabled(if: SystemLanguageModel.default.isAvailable))
            func aToolTakingInlineObjectsStartsASession() async throws {
                let session = FoundationModels.LanguageModelSession(tools: [PageTool(parameters: try Self.schema())])
                let response = try await session.respond(to: "hello!")
                #expect(!response.content.isEmpty)
            }
        #endif
    }
#endif
