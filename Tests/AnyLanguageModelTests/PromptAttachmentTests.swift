import Foundation
import Testing

@testable import AnyLanguageModel

#if canImport(CoreGraphics) && canImport(ImageIO)
    import CoreGraphics
    import ImageIO
#endif
#if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS) && !os(watchOS)
    import FoundationModels
#endif

@Suite("Prompt attachments")
struct PromptAttachmentTests {
    private let imageURL = URL(fileURLWithPath: "/tmp/attachment-fixture.png")

    private func prompt() -> AnyLanguageModel.Prompt {
        AnyLanguageModel.Prompt {
            "before"
            AnyLanguageModel.Attachment(imageURL: imageURL)
            "after"
        }
    }

    @Test func buildersAndRepresentablesPreserveOrderedImages() throws {
        let original = prompt()
        struct Representable: AnyLanguageModel.PromptRepresentable {
            let promptRepresentation: AnyLanguageModel.Prompt
        }
        let wrapped = AnyLanguageModel.Prompt(
            Representable(promptRepresentation: original)
        )
        let composed = AnyLanguageModel.Prompt {
            if true {
                wrapped
            }
            for item in [original] {
                item
            }
        }

        let expected = try original.makeTranscriptSegments()
        #expect(try wrapped.makeTranscriptSegments() == expected)
        let combined = try composed.makeTranscriptSegments()
        #expect(combined.count == 5)
        let sources: [AnyLanguageModel.Transcript.ImageSegment.Source] = combined.compactMap {
            if case .image(let value) = $0 {
                return value.source
            }
            return nil
        }
        let texts: [String] = combined.compactMap {
            if case .text(let value) = $0 {
                return value.content
            }
            return nil
        }
        #expect(sources == [.url(imageURL), .url(imageURL)])
        #expect(texts == ["before", "after\nbefore", "after"])
        #expect(original.description == "before\n<image>\nafter")
        #expect(
            try AnyLanguageModel.Prompt([original, original]).makeTranscriptSegments()
                == combined
        )
    }

    @Test func directSessionPromptsRetainImagesInBothModes() async throws {
        for streaming in [false, true] {
            let session = AnyLanguageModel.LanguageModelSession(model: MockLanguageModel())
            let input = prompt()
            if streaming {
                _ = try await session.streamResponse(to: input).collect()
            } else {
                _ = try await session.respond(to: input)
            }

            let first = try #require(session.transcript.first)
            guard case .prompt(let recorded) = first else {
                Issue.record("Missing prompt")
                return
            }
            #expect(recorded.segments == (try input.makeTranscriptSegments()))
        }
    }

    #if canImport(CoreGraphics) && canImport(ImageIO)
        private func image() throws -> CGImage {
            let context = try #require(
                CGContext(
                    data: nil,
                    width: 2,
                    height: 3,
                    bitsPerComponent: 8,
                    bytesPerRow: 8,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                )
            )
            return try #require(context.makeImage())
        }

        @Test func imageEncodingRetainsPixelsAndOrientation() throws {
            let value = try image()
            let input = AnyLanguageModel.Prompt {
                "image note"
                AnyLanguageModel.Attachment(value, orientation: .right)
            }
            let segments = try input.makeTranscriptSegments()
            guard case .image(let segment) = segments.last,
                case .data(let bytes, let mime) = segment.source
            else {
                Issue.record("No encoded image")
                return
            }
            #expect(mime == "image/png")
            let source = try #require(
                CGImageSourceCreateWithData(bytes as CFData, nil)
            )
            let decoded = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
            #expect(decoded.width == value.width)
            #expect(decoded.height == value.height)
            let properties = try #require(
                CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
            )
            #expect(
                (properties[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value
                    == CGImagePropertyOrientation.right.rawValue
            )
        }
    #endif

    #if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS) && !os(watchOS)
        @available(macOS 27, iOS 27, visionOS 27, *)
        @Test func nativeBridgeRetainsAttachment() async throws {
            let input = try AnyLanguageModel.Prompt {
                "before"
                AnyLanguageModel.Attachment(try image(), orientation: .right)
                "after"
            }
            let session = FoundationModels.LanguageModelSession(model: BridgeModel())
            _ = try await session.respond(to: input.toFoundationModels())
            guard case .prompt(let recorded) = try #require(session.transcript.first) else {
                Issue.record("Missing prompt")
                return
            }
            #expect(recorded.segments.count == 3)
            guard case .attachment(let segment) = recorded.segments[1],
                case .image(let image) = segment.content
            else {
                Issue.record("Native bridge lost attachment")
                return
            }
            #expect(image.orientation == .right)
        }

        @available(macOS 27, iOS 27, visionOS 27, *)
        private struct BridgeModel: FoundationModels.LanguageModel {
            typealias Executor = BridgeExecutor
            var capabilities: FoundationModels.LanguageModelCapabilities {
                .init([.vision])
            }
            var executorConfiguration: String {
                "bridge-fixture"
            }
        }

        @available(macOS 27, iOS 27, visionOS 27, *)
        private struct BridgeExecutor: FoundationModels.LanguageModelExecutor {
            init(configuration: String) {}

            func respond(
                to request: FoundationModels.LanguageModelExecutorGenerationRequest,
                model: BridgeModel,
                streamingInto channel: FoundationModels.LanguageModelExecutorGenerationChannel
            ) async throws {
                await channel.send(.response(action: .appendText("OK", tokenCount: 1)))
            }
        }
    #endif
}
