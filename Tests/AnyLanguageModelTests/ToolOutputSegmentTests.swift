import Foundation
import Testing

@_spi(Compatibility) @testable import AnyLanguageModel

#if canImport(UniformTypeIdentifiers)
    import UniformTypeIdentifiers
#endif

@Suite("Tool output segments")
struct ToolOutputSegmentTests {
    @Test func stringOutputIsPlainText() async throws {
        let output = "Sunny\nHigh of \"21\""
        let segments = try await StringTool(output: output).makeOutputSegments(
            from: GeneratedContent("read")
        )
        guard case .text(let text) = try #require(segments.first) else {
            Issue.record("Expected a text segment")
            return
        }
        #expect(segments.count == 1)
        #expect(text.content == output)
    }

    @Test func structuredOutputIsAStructure() async throws {
        let segments = try await StructuredTool().makeOutputSegments(from: GeneratedContent("read"))
        guard case .structure(let structure) = try #require(segments.first) else {
            Issue.record("Expected a structure segment")
            return
        }
        #expect(segments.count == 1)
        #expect(structure.source == "structured")
        #expect(structure.content == GeneratedContent(properties: ["count": 3]))
    }

    static let imageTypes: [(String, String)] = [
        ("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("png", "image/png"), ("gif", "image/gif"),
        ("webp", "image/webp"), ("tif", "image/tiff"), ("tiff", "image/tiff"), ("svg", "image/svg+xml"),
        ("bmp", "image/bmp"), ("heic", "image/heic"), ("heif", "image/heif"), ("avif", "image/avif"),
        ("ico", "image/vnd.microsoft.icon"),
    ]

    @Test(arguments: imageTypes)
    func fallbackImageTypes(fileExtension: String, mimeType: String) {
        #expect(toolOutputImageMIMEType(forPathExtension: fileExtension) == mimeType)
        #expect(toolOutputImageMIMEType(forPathExtension: fileExtension.uppercased()) == mimeType)
    }

    #if canImport(UniformTypeIdentifiers)
        /// The table for platforms without UniformTypeIdentifiers
        /// gives the same types as `UTType`.
        @Test(arguments: imageTypes)
        func fallbackImageTypesMatchUTType(fileExtension: String, mimeType: String) throws {
            let type = try #require(UTType(filenameExtension: fileExtension))
            #expect(type.conforms(to: .image))
            #expect(type.preferredMIMEType == mimeType)
        }
    #endif

    @Test(arguments: ["pdf", "txt", ""])
    func fallbackImageTypesExcludeOtherFiles(fileExtension: String) {
        #expect(toolOutputImageMIMEType(forPathExtension: fileExtension) == nil)
    }

    @Test(arguments: imageTypes)
    func localImageFilesGetTheirType(fileExtension: String, mimeType: String) async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(fileExtension)
        let data = Data("image bytes".utf8)
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let segments = try await PromptTool(output: Prompt { Attachment(imageURL: url) })
            .makeOutputSegments(from: GeneratedContent("read"))
        guard case .image(let image) = try #require(segments.first) else {
            Issue.record("Expected an image segment")
            return
        }
        #expect(image.source == .data(data, mimeType: mimeType))
    }
}

private struct StringTool: Tool {
    let name = "string"
    let description = "Return a string"
    let output: String

    var parameters: GenerationSchema {
        String.generationSchema
    }

    func call(arguments: String) async throws -> String {
        output
    }
}

private struct StructuredTool: Tool {
    let name = "structured"
    let description = "Return structured content"

    var parameters: GenerationSchema {
        String.generationSchema
    }

    func call(arguments: String) async throws -> GeneratedContent {
        GeneratedContent(properties: ["count": 3])
    }
}

private struct PromptTool: Tool {
    let name = "prompt"
    let description = "Return a prompt"
    let output: Prompt

    var parameters: GenerationSchema {
        String.generationSchema
    }

    func call(arguments: String) async throws -> Prompt {
        output
    }
}
