import Testing

#if canImport(FoundationModels) && !os(tvOS) && !os(watchOS)
    import FoundationModels
    @_spi(Compatibility) @testable import AnyLanguageModel

    @Suite("Native Tool session-property binding")
    struct NativeSessionPropertyTests {
        @Test func nativeWrappersBindPropertiesWithoutTaskLocalInheritance() async throws {
            guard #available(macOS 26.0, iOS 26.0, visionOS 26.0, *) else { return }
            let first = AnyLanguageModel.LanguageModelSession(model: MockLanguageModel(), tools: [NativePropertyTool()])
            let second = AnyLanguageModel.LanguageModelSession(
                model: MockLanguageModel(),
                tools: [NativePropertyTool()]
            )
            first.properties.revision = 1
            second.properties.revision = 20
            let firstTool = try #require(first.resolvedRequestContext().tools.toFoundationModels().first)
            let secondTool = try #require(second.resolvedRequestContext().tools.toFoundationModels().first)

            let firstCall = Task.detached { try await callNativeTool(firstTool) }
            let secondCall = Task.detached { try await callNativeTool(secondTool) }
            try await firstCall.value
            try await secondCall.value

            #expect(first.properties.revision == 2)
            #expect(second.properties.revision == 21)
            #expect(AnyLanguageModel.SessionPropertyBinding.values == nil)
        }
    }

    @available(macOS 26.0, iOS 26.0, visionOS 26.0, *)
    private func callNativeTool<T: FoundationModels.Tool>(_ tool: T) async throws {
        #expect(AnyLanguageModel.SessionPropertyBinding.values == nil)
        let arguments = try T.Arguments(FoundationModels.GeneratedContent(json: "\"read\""))
        _ = try await tool.call(arguments: arguments)
    }

    private struct NativePropertyTool: AnyLanguageModel.Tool {
        @AnyLanguageModel.LanguageModelSession.SessionProperty(\.revision) private var revision
        let name = "property"
        let description = "Update a session property"
        var parameters: AnyLanguageModel.GenerationSchema { String.generationSchema }

        func call(arguments: String) async throws -> String {
            revision += 1
            return "done"
        }
    }
#endif
