import Foundation

#if canImport(CoreGraphics) && canImport(ImageIO)
    import CoreGraphics
    import ImageIO
#endif

/// A typed prompt attachment, following Foundation Models 27's image surface.
///
/// AnyLanguageModel lowers image attachments into its existing compatibility
/// transcript image segments on deployment targets before OS 27.
public struct Attachment<Content> {
    let content: Content
}

/// Image content retained until a prompt is lowered into the compatibility transcript.
public struct ImageAttachmentContent: Sendable, Equatable {
    enum Source: Sendable {
        case url(URL, orientation: UInt32?)
        #if canImport(CoreGraphics) && canImport(ImageIO)
            case image(CGImage, orientation: CGImagePropertyOrientation?)
        #endif
    }

    let source: Source

    public static func == (lhs: Self, rhs: Self) -> Bool {
        switch (lhs.source, rhs.source) {
        case (.url(let a, let ao), .url(let b, let bo)):
            return a == b && ao == bo
        #if canImport(CoreGraphics) && canImport(ImageIO)
            case (.image(let a, let ao), .image(let b, let bo)):
                return a === b && ao == bo
        #endif
        default:
            return false
        }
    }

    func makeSegment(id: String) throws -> Transcript.ImageSegment {
        switch source {
        case .url(let url, let orientation):
            #if canImport(CoreGraphics) && canImport(ImageIO)
                if let orientation {
                    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
                    else {
                        throw Transcript.ImageEncodingError.imageConversionFailed
                    }
                    return try Self(
                        source: .image(
                            image,
                            orientation: CGImagePropertyOrientation(rawValue: orientation)
                        )
                    ).makeSegment(id: id)
                }
            #endif
            return .init(id: id, url: url)
        #if canImport(CoreGraphics) && canImport(ImageIO)
            case .image(let image, let orientation):
                let bytes = NSMutableData()
                guard
                    let destination = CGImageDestinationCreateWithData(
                        bytes,
                        "public.png" as CFString,
                        1,
                        nil
                    )
                else {
                    throw Transcript.ImageEncodingError.imageConversionFailed
                }
                var properties: [CFString: Any] = [:]
                if let orientation {
                    properties[kCGImagePropertyOrientation] = orientation.rawValue
                }
                CGImageDestinationAddImage(destination, image, properties as CFDictionary)
                guard CGImageDestinationFinalize(destination) else {
                    throw Transcript.ImageEncodingError.imageConversionFailed
                }
                return .init(id: id, data: bytes as Data, mimeType: "image/png")
        #endif
        }
    }
}

extension Attachment where Content == ImageAttachmentContent {
    #if canImport(CoreGraphics) && canImport(ImageIO)
        public init(_ image: CGImage, orientation: CGImagePropertyOrientation? = nil) {
            content = .init(source: .image(image, orientation: orientation))
        }

        public init(imageURL: URL, orientation: CGImagePropertyOrientation? = nil) {
            // Preserve the URL itself; its bytes are not read merely by constructing a prompt.
            content = .init(source: .url(imageURL, orientation: orientation?.rawValue))
        }
    #else
        public init(imageURL: URL) {
            content = .init(source: .url(imageURL, orientation: nil))
        }
    #endif
}

extension Attachment: PromptRepresentable where Content == ImageAttachmentContent {
    public var promptRepresentation: Prompt {
        Prompt(components: [.image(id: UUID().uuidString, content: content)])
    }
}
