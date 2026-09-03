import Foundation

/// A type that represents a conversation history between a user and a language model.
public struct Transcript: Sendable, Equatable, Codable {
    private var entries: [Entry]

    /// Creates a transcript.
    ///
    /// - Parameters:
    ///   - entries: An array of entries to seed the transcript.
    public init(entries: some Sequence<Entry> = []) {
        self.entries = Array(entries)
    }

    /// Appends a single entry to the transcript.
    mutating func append(_ entry: Entry) {
        entries.append(entry)
    }

    /// Appends multiple entries to the transcript.
    mutating func append<S>(contentsOf newEntries: S) where S: Sequence, S.Element == Entry {
        entries.append(contentsOf: newEntries)
    }
    
    mutating private func replace(index: Int, with entry: Entry) {
        entries[index] = entry
    }
    
    /// Updates a transcript with temporary text that is being streamed from a model.
    /// Appends the assistant response to the end of entries, if the last entry is a response then that response is updated with the newest streamed text.
    ///
    /// - Parameter text: The text to update the transcript with.
    mutating func appendStreamingResponse(_ text: String) {
        // Make sure the last entry in the transcript is a response. If it is not, create a new response and append it to the end of the transcript.
        guard case .response(var response) = entries.last else {
            append(Entry.response(Response(assetIDs: [], segments: [
                Transcript.Segment.text(Transcript.TextSegment(content: text))
            ])))
            return
        }
        
        // If the last segment in the last response is text, replace it with the new content.
        if case .text(let last)? = response.segments.last {
            // Keep the same ID as the last segment.
            response.segments[response.segments.count - 1] = Transcript.Segment.text(Transcript.TextSegment(id: last.id, content: text))
        } else {
            response.segments.append(Transcript.Segment.text(Transcript.TextSegment(content: text)))
        }
        
        // Replace the latest entry with the one we just updated.
        replace(index: entries.count - 1, with: .response(response))
    }
    
    /// Replaces the trailing response entry's text with the final text, or appends a new response entry if the last entry isn't a response.
    /// Prevents streamed responses from having duplicate entries on completion.
    ///
    /// - Parameters:
    ///   - text: The text to replace the final response with.
    ///   - assetIDs: The assetIDs for the response.
    mutating func finalizeStreamedTranscript(_ text: String, assetIDs: [String]) {
        // Make sure the last entry in the transcript is a response. If it is not, create a new response and append it to the end of the transcript.
        guard case .response(let response) = entries.last else {
            append(Entry.response(Response(assetIDs: assetIDs, segments: [
                Transcript.Segment.text(Transcript.TextSegment(content: text))
            ])))
            return
        }
               
        // If the last segment is text we want to carry its ID over to the new text segment. Otherwise generate a new ID for it.
        let id = switch response.segments.last {
        case .text(let last):
            last.id
        default:
            UUID().uuidString
        }
        
        let newResponse: Entry = Entry.response(Response(id: response.id, assetIDs: assetIDs, segments: [
            Transcript.Segment.text(Transcript.TextSegment(id: id, content: text))
        ]))
        
        replace(index: entries.count - 1, with: newResponse)
    }


    /// An entry in a transcript.
    public enum Entry: Sendable, Identifiable, Equatable, Codable {
        /// Instructions, typically provided by you, the developer.
        case instructions(Instructions)

        /// A prompt, typically sourced from an end user.
        case prompt(Prompt)

        /// A tool call containing a tool name and the arguments to invoke it with.
        case toolCalls(ToolCalls)

        /// An tool output provided back to the model.
        case toolOutput(ToolOutput)

        /// A response from the model.
        case response(Response)

        /// The stable identity of the entity associated with this instance.
        public var id: String {
            switch self {
            case .instructions(let instructions):
                return instructions.id
            case .prompt(let prompt):
                return prompt.id
            case .toolCalls(let toolCalls):
                return toolCalls.id
            case .toolOutput(let toolOutput):
                return toolOutput.id
            case .response(let response):
                return response.id
            }
        }
    }

    /// The types of segments that may be included in a transcript entry.
    public enum Segment: Sendable, Identifiable, Equatable, Codable {
        /// A segment containing text.
        case text(TextSegment)

        /// A segment containing structured content.
        case structure(StructuredSegment)

        /// A segment containing an image.
        case image(ImageSegment)

        /// The stable identity of the entity associated with this instance.
        public var id: String {
            switch self {
            case .text(let textSegment):
                return textSegment.id
            case .structure(let structuredSegment):
                return structuredSegment.id
            case .image(let imageSegment):
                return imageSegment.id
            }
        }
    }

    /// A segment containing text.
    public struct TextSegment: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        public var content: String

        public init(id: String = UUID().uuidString, content: String) {
            self.id = id
            self.content = content
        }
    }

    /// A segment containing structured content.
    public struct StructuredSegment: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        /// A source that be used to understand which type content represents.
        public var source: String

        /// The content of the segment.
        public var content: GeneratedContent

        public init(id: String = UUID().uuidString, source: String, content: GeneratedContent) {
            self.id = id
            self.source = source
            self.content = content
        }
    }

    /// A segment that represents an image for multi‑modal prompts and outputs.
    ///
    /// Use this type to include images alongside text and structured content when
    /// constructing `Transcript` entries. Images can be provided as raw data with a
    /// MIME type or by URL.
    public struct ImageSegment: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        /// The source of the image data.
        public let source: Source

        /// The origin of an image's content.
        public enum Source: Sendable, Equatable, Codable {
            /// Image bytes and their MIME type (for example, `image/jpeg`).
            case data(Data, mimeType: String)
            /// A URL that references an image.
            case url(URL)

            private enum CodingKeys: String, CodingKey { case kind, data, mimeType, url }

            public init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                let kind = try container.decode(String.self, forKey: .kind)
                switch kind {
                case "data":
                    let data = try container.decode(Data.self, forKey: .data)
                    let mimeType = try container.decode(String.self, forKey: .mimeType)
                    self = .data(data, mimeType: mimeType)
                case "url":
                    let url = try container.decode(URL.self, forKey: .url)
                    self = .url(url)
                default:
                    throw DecodingError.dataCorrupted(
                        .init(codingPath: [CodingKeys.kind], debugDescription: "Unknown image source kind: \(kind)")
                    )
                }
            }

            public func encode(to encoder: Encoder) throws {
                var container = encoder.container(keyedBy: CodingKeys.self)
                switch self {
                case .data(let data, let mimeType):
                    try container.encode("data", forKey: .kind)
                    try container.encode(data, forKey: .data)
                    try container.encode(mimeType, forKey: .mimeType)
                case .url(let url):
                    try container.encode("url", forKey: .kind)
                    try container.encode(url, forKey: .url)
                }
            }
        }

        /// Creates an image segment from a source.
        ///
        /// - Parameters:
        ///   - id: A unique identifier for this segment. Defaults to a generated UUID.
        ///   - source: The image source.
        public init(id: String = UUID().uuidString, source: Source) {
            self.id = id
            self.source = source
        }

        /// Creates an image segment from raw bytes.
        ///
        /// - Parameters:
        ///   - id: A unique identifier for this segment. Defaults to a generated UUID.
        ///   - data: The encoded image bytes.
        ///   - mimeType: The MIME type corresponding to the image data (for example, `image/png`).
        public init(id: String = UUID().uuidString, data: Data, mimeType: String) {
            self.id = id
            self.source = .data(data, mimeType: mimeType)
        }

        /// Creates an image segment from a URL.
        ///
        /// - Parameters:
        ///   - id: A unique identifier for this segment. Defaults to a generated UUID.
        ///   - url: A URL that references an image.
        public init(id: String = UUID().uuidString, url: URL) {
            self.id = id
            self.source = .url(url)
        }
    }

    /// Errors that can occur when converting platform images to encoded data.
    public enum ImageEncodingError: Error {
        /// The image couldn't be converted to the requested format.
        case imageConversionFailed
    }

    /// Instructions you provide to the model that define its behavior.
    ///
    /// Instructions are typically provided to define the role and behavior of the model. Apple trains the model
    /// to obey instructions over any commands it receives in prompts. This is a security mechanism to help
    /// mitigate prompt injection attacks.
    public struct Instructions: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        /// The content of the instructions, in natural language.
        ///
        /// - Note: Instructions are often provided in English even when the
        /// users interact with the model in another language.
        public var segments: [Segment]

        /// A list of tools made available to the model.
        public var toolDefinitions: [ToolDefinition]

        /// Initialize instructions by describing how you want the model to
        /// behave using natural language.
        ///
        /// - Parameters:
        ///   - id: A unique identifier for this instructions segment.
        ///   - segments: An array of segments that make up the instructions.
        ///   - toolDefinitions: Tools that the model should be allowed to call.
        public init(
            id: String = UUID().uuidString,
            segments: [Segment],
            toolDefinitions: [ToolDefinition]
        ) {
            self.id = id
            self.segments = segments
            self.toolDefinitions = toolDefinitions
        }
    }

    /// A prompt from the user asking the model.
    public struct Prompt: Sendable, Identifiable, Equatable, Codable {
        /// The identifier of the prompt.
        public var id: String

        /// Ordered prompt segments.
        public var segments: [Segment]

        /// Generation options associated with the prompt.
        public var options: GenerationOptions

        /// An optional response format that describes the desired output structure.
        public var responseFormat: ResponseFormat?

        /// Creates a prompt.
        ///
        /// - Parameters:
        ///   - id: A ``Generable`` type to use as the response format.
        ///   - segments: An array of segments that make up the prompt.
        ///   - options: Options that control how tokens are sampled from the distribution the model produces.
        ///   - responseFormat: A response format that describes the output structure.
        public init(
            id: String = UUID().uuidString,
            segments: [Segment],
            options: GenerationOptions = GenerationOptions(),
            responseFormat: ResponseFormat? = nil
        ) {
            self.id = id
            self.segments = segments
            self.options = options
            self.responseFormat = responseFormat
        }
    }

    /// Specifies a response format that the model must conform its output to.
    public struct ResponseFormat: Sendable, Codable {
        private let schema: GenerationSchema

        /// A name associated with the response format.
        public var name: String {
            // Extract type name from the schema's debug description
            // This is a best-effort approach
            let desc = schema.debugDescription
            if let range = desc.range(of: "$ref("),
                let endRange = desc.range(of: ")", range: range.upperBound ..< desc.endIndex)
            {
                let name = desc[range.upperBound ..< endRange.lowerBound]
                return String(name)
            }
            return "response"
        }

        /// Creates a response format with type you specify.
        ///
        /// - Parameters:
        ///   - type: A ``Generable`` type to use as the response format.
        public init<Content>(type: Content.Type) where Content: Generable {
            self.schema = Content.generationSchema
        }

        /// Creates a response format with a schema.
        ///
        /// - Parameters:
        ///   - schema: A schema to use as the response format.
        public init(schema: GenerationSchema) {
            self.schema = schema
        }
    }

    /// A collection tool calls generated by the model.
    public struct ToolCalls: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        private var calls: [ToolCall]

        public init<S>(id: String = UUID().uuidString, _ calls: S)
        where S: Sequence, S.Element == ToolCall {
            self.id = id
            self.calls = Array(calls)
        }
    }

    /// A tool call generated by the model containing the name of a tool and arguments to pass to it.
    public struct ToolCall: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        /// The name of the tool being invoked.
        public var toolName: String

        /// Arguments to pass to the invoked tool.
        public var arguments: GeneratedContent

        public init(id: String, toolName: String, arguments: GeneratedContent) {
            self.id = id
            self.toolName = toolName
            self.arguments = arguments
        }
    }

    /// A tool output provided back to the model.
    public struct ToolOutput: Sendable, Identifiable, Equatable, Codable {
        /// A unique id for this tool output.
        public var id: String

        /// The name of the tool that produced this output.
        public var toolName: String

        /// Segments of the tool output.
        public var segments: [Segment]

        public init(id: String, toolName: String, segments: [Segment]) {
            self.id = id
            self.toolName = toolName
            self.segments = segments
        }
    }

    /// A response from the model.
    public struct Response: Sendable, Identifiable, Equatable, Codable {
        /// The stable identity of the entity associated with this instance.
        public var id: String

        /// Version aware identifiers for all assets used to generate this response.
        public var assetIDs: [String]

        /// Ordered prompt segments.
        public var segments: [Segment]

        public init(
            id: String = UUID().uuidString,
            assetIDs: [String],
            segments: [Segment]
        ) {
            self.id = id
            self.assetIDs = assetIDs
            self.segments = segments
        }
    }

    /// A definition of a tool.
    public struct ToolDefinition: Sendable, Codable {
        /// The tool's name.
        public var name: String

        /// A description of how and when to use the tool.
        public var description: String

        /// The schema describing the tool's parameters.
        internal let parameters: GenerationSchema

        public init(name: String, description: String, parameters: GenerationSchema) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }

        public init(tool: some Tool) {
            self.name = tool.name
            self.description = tool.description
            self.parameters = tool.parameters
        }
    }
}

// MARK: - CustomStringConvertible

extension Transcript.Entry: CustomStringConvertible {
    public var description: String {
        switch self {
        case .instructions(let instructions):
            return "instructions(\(instructions))"
        case .prompt(let prompt):
            return "prompt(\(prompt))"
        case .toolCalls(let toolCalls):
            return "toolCalls(\(toolCalls))"
        case .toolOutput(let toolOutput):
            return "toolOutput(\(toolOutput))"
        case .response(let response):
            return "response(\(response))"
        }
    }
}

extension Transcript.Segment: CustomStringConvertible {
    public var description: String {
        switch self {
        case .text(let textSegment):
            return textSegment.description
        case .structure(let structuredSegment):
            return structuredSegment.description
        case .image:
            return "<image>"
        }
    }
}

extension Transcript.TextSegment: CustomStringConvertible {
    public var description: String { content }
}

extension Transcript.StructuredSegment: CustomStringConvertible {
    public var description: String {
        "StructuredSegment(source: \(source), content: \(content))"
    }
}

extension Transcript.ImageSegment {
    /// Preferred image encodings for image conversion.
    public enum Format: Sendable {
        /// JPEG encoding with the specified compression quality.
        case jpeg(compressionQuality: Double = 0.9)
        /// PNG encoding.
        case png
    }
}

#if canImport(UIKit) || targetEnvironment(macCatalyst)
    import UIKit

    extension Transcript.ImageSegment {
        fileprivate static func encode(_ image: UIImage, format: Format) throws -> (Data, String) {
            switch format {
            case .jpeg(let quality):
                guard let data = image.jpegData(compressionQuality: quality) else {
                    throw Transcript.ImageEncodingError.imageConversionFailed
                }
                return (data, "image/jpeg")
            case .png:
                guard let data = image.pngData() else {
                    throw Transcript.ImageEncodingError.imageConversionFailed
                }
                return (data, "image/png")
            }
        }

        /// Creates an image segment by encoding a UIKit image.
        ///
        /// - Parameters:
        ///   - image: The source image to encode.
        ///   - format: The target encoding. Defaults to JPEG with 0.9 quality.
        /// - Throws: ``Transcript/ImageEncodingError-swift.enum/imageConversionFailed`` if encoding fails.
        public init(image: UIImage, format: Format = .jpeg()) throws {
            let (data, mimeType) = try Self.encode(image, format: format)
            self.init(data: data, mimeType: mimeType)
        }
    }
#elseif canImport(AppKit)
    import AppKit

    extension Transcript.ImageSegment {
        fileprivate static func encode(_ image: NSImage, format: Format) throws -> (Data, String) {
            guard let tiffData = image.tiffRepresentation,
                let bitmapImage = NSBitmapImageRep(data: tiffData)
            else {
                throw Transcript.ImageEncodingError.imageConversionFailed
            }

            switch format {
            case .jpeg(let quality):
                guard
                    let data = bitmapImage.representation(
                        using: .jpeg,
                        properties: [.compressionFactor: quality]
                    )
                else {
                    throw Transcript.ImageEncodingError.imageConversionFailed
                }
                return (data, "image/jpeg")
            case .png:
                guard
                    let data = bitmapImage.representation(
                        using: .png,
                        properties: [:]
                    )
                else {
                    throw Transcript.ImageEncodingError.imageConversionFailed
                }
                return (data, "image/png")
            }
        }

        /// Creates an image segment by encoding an AppKit image.
        ///
        /// - Parameters:
        ///   - image: The source image to encode.
        ///   - format: The target encoding. Defaults to JPEG with 0.9 quality.
        /// - Throws: ``Transcript/ImageEncodingError-swift.enum/imageConversionFailed`` if encoding fails.
        public init(image: NSImage, format: Format = .jpeg()) throws {
            let (data, mimeType) = try Self.encode(image, format: format)
            self.init(data: data, mimeType: mimeType)
        }
    }
#endif

#if canImport(CoreGraphics)
    import CoreGraphics
    import ImageIO
    import UniformTypeIdentifiers

    extension Transcript.ImageSegment {
        fileprivate static func encode(_ image: CGImage, format: Format) throws -> (Data, String) {
            let data = NSMutableData()
            let utType: UTType

            switch format {
            case .jpeg:
                utType = .jpeg
            case .png:
                utType = .png
            }

            guard
                let destination = CGImageDestinationCreateWithData(
                    data,
                    utType.identifier as CFString,
                    1,
                    nil
                )
            else {
                throw Transcript.ImageEncodingError.imageConversionFailed
            }

            var properties: [CFString: Any] = [:]
            if case .jpeg(let quality) = format {
                properties[kCGImageDestinationLossyCompressionQuality] = quality
            }

            CGImageDestinationAddImage(destination, image, properties as CFDictionary)

            guard CGImageDestinationFinalize(destination) else {
                throw Transcript.ImageEncodingError.imageConversionFailed
            }

            let mimeType = (utType == .jpeg) ? "image/jpeg" : "image/png"
            return (data as Data, mimeType)
        }

        /// Creates an image segment by encoding a CoreGraphics image.
        ///
        /// - Parameters:
        ///   - image: The source image to encode.
        ///   - format: The target encoding. Defaults to JPEG with 0.9 quality.
        /// - Throws: ``Transcript/ImageEncodingError-swift.enum/imageConversionFailed`` if encoding fails.
        public init(image: CGImage, format: Format = .jpeg()) throws {
            let (data, mimeType) = try Self.encode(image, format: format)
            self.init(data: data, mimeType: mimeType)
        }
    }
#endif

extension Transcript.Instructions: CustomStringConvertible {
    public var description: String {
        "Instructions(segments: \(segments.count), tools: \(toolDefinitions.count))"
    }
}

extension Transcript.Prompt: CustomStringConvertible {
    public var description: String {
        "Prompt(segments: \(segments.count))"
    }
}

extension Transcript.ResponseFormat: CustomStringConvertible {
    public var description: String {
        "ResponseFormat(name: \(name))"
    }
}

extension Transcript.ToolCalls: CustomStringConvertible {
    public var description: String {
        "ToolCalls(\(count) calls)"
    }
}

extension Transcript.ToolCall: CustomStringConvertible {
    public var description: String {
        "ToolCall(tool: \(toolName))"
    }
}

extension Transcript.ToolOutput: CustomStringConvertible {
    public var description: String {
        "ToolOutput(tool: \(toolName), segments: \(segments.count))"
    }
}

extension Transcript.Response: CustomStringConvertible {
    public var description: String {
        "Response(segments: \(segments.count))"
    }
}

// MARK: - Equatable

extension Transcript.ResponseFormat: Equatable {
    public static func == (lhs: Transcript.ResponseFormat, rhs: Transcript.ResponseFormat) -> Bool {
        return lhs.name == rhs.name
    }
}

extension Transcript.ToolDefinition: Equatable {
    public static func == (lhs: Transcript.ToolDefinition, rhs: Transcript.ToolDefinition) -> Bool {
        return lhs.name == rhs.name && lhs.description == rhs.description
    }
}

// MARK: - RandomAccessCollection

extension Transcript: RandomAccessCollection {
    public subscript(index: Int) -> Entry {
        entries[index]
    }

    public var startIndex: Int {
        entries.startIndex
    }

    public var endIndex: Int {
        entries.endIndex
    }
}

extension Transcript.ToolCalls: RandomAccessCollection {
    public subscript(position: Int) -> Transcript.ToolCall {
        calls[position]
    }

    public var startIndex: Int {
        calls.startIndex
    }

    public var endIndex: Int {
        calls.endIndex
    }
}
