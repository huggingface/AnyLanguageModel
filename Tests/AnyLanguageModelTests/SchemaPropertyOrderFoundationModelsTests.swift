import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(FoundationModels)
    import FoundationModels

    @Suite("Schema property order for Foundation Models")
    struct SchemaPropertyOrderFoundationModelsTests {
        /// Foundation Models generates properties in the order its schema's `x-order` names.
        @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
        @Test func foundationModelsSchemaKeepsDeclarationOrder() throws {
            let schema = FoundationModels.GenerationSchema(OrderedItem.generationSchema)
            let json = String(decoding: try JSONEncoder().encode(schema), as: UTF8.self)
            #expect(json.contains(#""x-order":["kind","words","items"]"#))
            #expect(json.contains(#""x-order":["kind","words"]"#))
        }
    }
#endif
