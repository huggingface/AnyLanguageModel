import Testing

#if canImport(FoundationModels) && !os(watchOS)
    import AnyLanguageModel

    private let isSystemLanguageModelAvailable: Bool = {
        if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            return SystemLanguageModel.default.isAvailable
        }
        return false
    }()

    @Test func anyLanguageModelHistoryViewCollectionCompatibility() {
        func requireContract<C>(_ history: C)
        where
            C: RandomAccessCollection & MutableCollection & RangeReplaceableCollection
                & ExpressibleByArrayLiteral & Sendable,
            C.Element == Transcript.Entry, C.SubSequence == C, C.Index: Hashable & Strideable,
            C.Index.Stride == Int
        {}
        var history: Transcript.HistoryView = []
        let empty: Transcript.HistoryView = history[history.startIndex ..< history.endIndex]
        history[history.startIndex ..< history.endIndex] = empty
        history.append(contentsOf: empty)
        history.replaceSubrange(history.startIndex ..< history.endIndex, with: empty)
        requireContract(history)
        #expect(history.isEmpty)
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    extension SessionPropertyValues {
        @SessionPropertyEntry var compatibilityRevision = 0
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private struct CompatibilityDynamicInstructions: DynamicInstructions {
        let includeDetail: Bool
        @SessionProperty(\.compatibilityRevision) private var revision

        var body: some DynamicInstructions {
            Instructions("You are a helpful assistant. Revision \(revision).")
            if includeDetail {
                Instructions("Include useful detail.")
            }
        }
    }

    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
    @Test("AnyLanguageModel Drop-In Compatibility", .enabled(if: isSystemLanguageModelAvailable))
    func anyLanguageModelCompatibility() async throws {
        let model = SystemLanguageModel.default
        let session = LanguageModelSession(
            model: model,
            instructions: Instructions("You are a helpful assistant.")
        )

        if #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) {
            let dynamicSession = LanguageModelSession(
                model: model,
                dynamicInstructions: CompatibilityDynamicInstructions(includeDetail: true),
                history: session.transcript
            )
            dynamicSession.properties.compatibilityRevision = 1
            _ = dynamicSession.properties.history
        }

        let options = GenerationOptions(temperature: 0.7)
        let response = try await session.respond(options: options) {
            Prompt("Say 'Hello'")
        }
        #expect(!response.content.isEmpty)

        let stream = session.streamResponse {
            Prompt("Count to 3")
        }
        var hasSnapshots = false
        for try await _ in stream {
            hasSnapshots = true
            break
        }
        #expect(hasSnapshots)
    }
#endif
