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
    private enum SegmentPayload: Equatable {
        case text(String)
        case image(AnyLanguageModel.Transcript.ImageSegment.Source)
    }

    private let imageURL = URL(fileURLWithPath: "/tmp/attachment-fixture.png")

    private func payloads(
        _ segments: [AnyLanguageModel.Transcript.Segment]
    ) -> [SegmentPayload] {
        segments.compactMap { segment in
            switch segment {
            case .text(let text):
                .text(text.content)
            case .image(let image):
                .image(image.source)
            case .structure:
                nil
            }
        }
    }

    private func prompt() -> AnyLanguageModel.Prompt {
        AnyLanguageModel.Prompt {
            "before"
            AnyLanguageModel.Attachment(imageURL: imageURL)
            "after"
        }
    }

    @Test func emptyBuildersKeepOneEmptyTextSegment() throws {
        let built = AnyLanguageModel.Prompt {}
        let array = AnyLanguageModel.Prompt([AnyLanguageModel.Prompt]())

        for prompt in [built, array] {
            let segments = try prompt.makeTranscriptSegments()
            #expect(segments.count == 1)
            guard case .text(let text) = segments[0] else {
                Issue.record("Empty prompt did not retain its text segment")
                continue
            }
            #expect(text.content.isEmpty)
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
        let wrappedSegments = try wrapped.makeTranscriptSegments()
        #expect(payloads(wrappedSegments) == payloads(expected))
        #expect(Set(wrappedSegments.map(\.id)).isDisjoint(with: expected.map(\.id)))
        let combined = try composed.makeTranscriptSegments()
        #expect(combined.count == 5)
        #expect(Set(combined.map(\.id)).count == combined.count)
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
        let repeated = try AnyLanguageModel.Prompt([original, original]).makeTranscriptSegments()
        #expect(payloads(repeated) == payloads(combined))
        #expect(Set(repeated.map(\.id)).count == repeated.count)
        #expect(Set(repeated.map(\.id)).isDisjoint(with: combined.map(\.id)))
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
            #expect(payloads(recorded.segments) == payloads(try input.makeTranscriptSegments()))
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

        #if canImport(FoundationModels)
            @available(macOS 26.0, iOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
            @Test func duplicateSuppressionDoesNotReencodeRecordedAttachments() throws {
                let encoded = try AnyLanguageModel.Prompt {
                    AnyLanguageModel.Attachment(try image(), orientation: .right)
                }.makeTranscriptSegments()
                guard case .image(let image) = try #require(encoded.first),
                    case .data(let data, _) = image.source
                else {
                    Issue.record("Failed to encode the image fixture")
                    return
                }

                let url = FileManager.default.temporaryDirectory
                    .appending(path: UUID().uuidString)
                    .appendingPathExtension("png")
                try data.write(to: url)
                let prompt = AnyLanguageModel.Prompt {
                    AnyLanguageModel.Attachment(imageURL: url, orientation: .right)
                }
                let occurrenceID = UUID().uuidString
                let segments = try prompt.makeTranscriptSegments(occurrenceID: occurrenceID)
                try FileManager.default.removeItem(at: url)
                let transcript = AnyLanguageModel.Transcript(entries: [
                    .prompt(.init(id: occurrenceID, segments: segments))
                ])

                #expect(
                    AnyLanguageModel.fmTranscriptDroppingDuplicatePrompt(
                        transcript,
                        prompt: prompt
                    ).isEmpty
                )
            }
        #endif
    #endif

    #if canImport(FoundationModels)
        @Test func bridgeErrorsDescribeTheirCause() {
            let unavailable = AnyLanguageModel.PromptBridgeError.attachmentsUnavailable
            let nonFileURL = AnyLanguageModel.PromptBridgeError.nonFileImageURL
            #expect(unavailable.localizedDescription.contains("Swift 6.4"))
            #expect(unavailable.localizedDescription.contains("OS 27"))
            #expect(nonFileURL.localizedDescription.contains("local files"))
        }
    #endif

    #if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS) && !os(watchOS)
        @available(macOS 27, iOS 27, visionOS 27, *)
        @Test func nativeBridgeDescribesRemoteImageURLs() throws {
            let url = try #require(URL(string: "https://example.com/image.png"))
            let input = AnyLanguageModel.Prompt {
                AnyLanguageModel.Attachment(imageURL: url)
            }
            #expect {
                _ = try input.toFoundationModels()
            } throws: { error in
                error.localizedDescription.contains("local files")
            }
        }

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
        @Test func foundationWrapperDoesNotDuplicatePromptsAndReplaysOrientation() async throws {
            let model = AnyLanguageModel.FoundationLanguageModel(BridgeModel())
            let session = AnyLanguageModel.LanguageModelSession(model: model)
            let input = try AnyLanguageModel.Prompt {
                "before"
                AnyLanguageModel.Attachment(try image(), orientation: .right)
                "after"
            }
            let first = try await session.respond(to: input)
            #expect(first.content == "prompts:1 images:1 right:1")

            let second = try await session.respond(to: "follow-up")
            #expect(second.content == "prompts:2 images:1 right:1")
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
                let prompts = request.transcript.compactMap { entry -> FoundationModels.Transcript.Prompt? in
                    guard case .prompt(let prompt) = entry else { return nil }
                    return prompt
                }
                let images = prompts.flatMap { prompt in
                    prompt.segments.compactMap {
                        segment -> FoundationModels.Transcript.ImageAttachment? in
                        guard case .attachment(let attachment) = segment,
                            case .image(let image) = attachment.content
                        else { return nil }
                        return image
                    }
                }
                let right = images.count { $0.orientation == .right }
                await channel.send(
                    .response(
                        action: .appendText(
                            "prompts:\(prompts.count) images:\(images.count) right:\(right)",
                            tokenCount: 1
                        )
                    )
                )
            }
        }
    #endif
}
