import Testing

@testable import AnyLanguageModel

private struct ToolSchemaOptOutTool: Tool {
    let name = "schemaOptOutTool"
    let description = "A tool that opts out of schema injection."
    let includesSchemaInInstructions = false

    @Generable
    struct Arguments {
        @Guide(description: "A value to echo")
        var value: String
    }

    func call(arguments: Arguments) async throws -> String {
        arguments.value
    }
}

private func waitUntil(
    timeout: Duration = .milliseconds(500),
    pollInterval: Duration = .milliseconds(5),
    _ condition: @autoclosure () -> Bool
) async throws -> Bool {
    let deadline = ContinuousClock.now + timeout
    while !condition() {
        if ContinuousClock.now >= deadline {
            return false
        }
        try await Task.sleep(for: pollInterval)
    }
    return true
}

@Suite("MockLanguageModel")
struct MockLanguageModelTests {
    @Test func fixedResponse() async throws {
        let model = MockLanguageModel.fixed("Hello, World!")
        let session = LanguageModelSession(model: model)

        #expect(session.transcript.count == 0)

        let response = try await session.respond(to: "Say hello")
        #expect(response.content == "Hello, World!")

        // Verify transcript was updated (prompt + response)
        #expect(session.transcript.count == 2)
    }

    @Test func echoResponse() async throws {
        let model = MockLanguageModel.echo
        let session = LanguageModelSession(model: model)

        let prompt = Prompt("Echo this")
        let response = try await session.respond(to: prompt)
        #expect(response.content.contains(prompt.description))

        // Verify transcript
        #expect(session.transcript.count == 2)
    }

    @Test func withInstructions() async throws {
        let model = MockLanguageModel { prompt, _ in
            if prompt.description.contains("Be helpful") {
                return "😇"
            }

            if prompt.description.contains("Be evil") {
                return "😈"
            }

            return "😐"
        }

        for (instructionText, expected) in [
            ("Be helpful", "😇"),
            ("Be evil", "😈"),
            ("Meh", "😐"),
        ] {
            let session = LanguageModelSession(
                model: model,
                instructions: Instructions(instructionText)
            )

            // Verify instructions are in transcript
            let entriesBeforeResponse = Array(session.transcript)
            #expect(entriesBeforeResponse.count == 1)
            if case .instructions(let transcriptInstructions) = entriesBeforeResponse.first {
                #expect(transcriptInstructions.segments.count > 0)
            } else {
                Issue.record("First entry should be instructions")
            }

            let response = try await session.respond(to: "Do what you want")
            #expect(response.content == expected)

            // Verify transcript has instructions, prompt, and response
            #expect(session.transcript.count == 3)
        }
    }

    @Test func instructionsOnlyIncludeToolsThatOptInToSchemaInjection() {
        let model = MockLanguageModel.fixed("ok")
        let tools: [any Tool] = [WeatherTool(), ToolSchemaOptOutTool()]
        let session = LanguageModelSession(
            model: model,
            tools: tools,
            instructions: "Use tools when appropriate."
        )

        #expect(session.transcript.count == 1)
        guard case let .instructions(instructionsEntry) = session.transcript[0] else {
            Issue.record("First entry should be instructions")
            return
        }

        #expect(instructionsEntry.toolDefinitions.count == 1)
        #expect(instructionsEntry.toolDefinitions.first?.name == "getWeather")
    }

    @Test func transcriptAndInstructionsInitHaveEquivalentToolDefinitionBehavior() {
        let model = MockLanguageModel.fixed("ok")
        let tools: [any Tool] = [WeatherTool(), ToolSchemaOptOutTool()]
        let expectedToolDefinitions =
            tools
            .filter(\.includesSchemaInInstructions)
            .map { Transcript.ToolDefinition(tool: $0) }

        let instructionsSession = LanguageModelSession(
            model: model,
            tools: tools,
            instructions: "Use tools when appropriate."
        )

        let explicitTranscript = Transcript(entries: [
            .instructions(
                Transcript.Instructions(
                    segments: [.text(.init(content: "Use tools when appropriate."))],
                    toolDefinitions: expectedToolDefinitions
                )
            )
        ])
        let transcriptSession = LanguageModelSession(
            model: model,
            tools: tools,
            transcript: explicitTranscript
        )

        guard case let .instructions(instructionsEntry) = instructionsSession.transcript[0] else {
            Issue.record("First instructions-based entry should be instructions")
            return
        }
        guard case let .instructions(transcriptEntry) = transcriptSession.transcript[0] else {
            Issue.record("First transcript-based entry should be instructions")
            return
        }

        #expect(instructionsEntry.toolDefinitions == transcriptEntry.toolDefinitions)
    }

    @Test func explicitTranscriptInstructionsArePreservedWhenToolsOptOut() {
        let model = MockLanguageModel.fixed("ok")
        let tools: [any Tool] = [WeatherTool(), ToolSchemaOptOutTool()]
        let explicitToolDefinitions = tools.map { Transcript.ToolDefinition(tool: $0) }
        let transcript = Transcript(entries: [
            .instructions(
                Transcript.Instructions(
                    segments: [.text(.init(content: "Custom instructions."))],
                    toolDefinitions: explicitToolDefinitions
                )
            )
        ])

        let session = LanguageModelSession(
            model: model,
            tools: tools,
            transcript: transcript
        )

        guard case let .instructions(entry) = session.transcript[0] else {
            Issue.record("First entry should be instructions")
            return
        }

        #expect(entry.toolDefinitions == explicitToolDefinitions)
    }

    @Test func unavailable() async throws {
        let model = MockLanguageModel.unavailable

        #expect(model.availability == .unavailable(.custom("MockLanguageModel is unavailable")))
        #expect(model.isAvailable == false)
    }

    @Test func streamingResponse() async throws {
        // Test async response with isResponding state
        let asyncModel = MockLanguageModel { _, _ in
            try await Task.sleep(for: .milliseconds(100))
            return "Async Response"
        }
        let asyncSession = LanguageModelSession(model: asyncModel)

        #expect(asyncSession.isResponding == false)
        #expect(asyncSession.transcript.count == 0)

        let asyncTask = Task {
            try await asyncSession.respond(to: "Async test")
        }

        #expect(try await waitUntil(asyncSession.isResponding == true))

        _ = try await asyncTask.value
        try await Task.sleep(for: .milliseconds(10))
        #expect(asyncSession.isResponding == false)
        #expect(asyncSession.transcript.count == 2)

        // Test streaming response with isResponding state
        let streamModel = MockLanguageModel.streamingMock()
        let streamSession = LanguageModelSession(model: streamModel)

        #expect(streamSession.isResponding == false)
        #expect(streamSession.transcript.count == 0)

        let stream = streamSession.streamResponse(to: "Stream test")

        let streamTask = Task {
            for try await _ in stream {}
        }

        #expect(try await waitUntil(streamSession.isResponding == true))

        _ = try await streamTask.value
        try await Task.sleep(for: .milliseconds(10))
        #expect(streamSession.isResponding == false)
        #expect(streamSession.transcript.count == 2)
    }

    @Test func transcriptGrowsWithMultipleInteractions() async throws {
        let model = MockLanguageModel.echo
        let session = LanguageModelSession(model: model)

        #expect(session.transcript.count == 0)

        try await session.respond(to: "First prompt")
        let countAfterFirst = session.transcript.count
        #expect(countAfterFirst == 2)

        try await session.respond(to: "Second prompt")
        let countAfterSecond = session.transcript.count
        #expect(countAfterSecond == 4)

        try await session.respond(to: "Third prompt")
        let countAfterThird = session.transcript.count
        #expect(countAfterThird == 6)

        // Verify all entries are identifiable
        for entry in session.transcript {
            #expect(!entry.id.isEmpty)
        }
    }

    @Test func respondWithSingleImage() async throws {
        let model = MockLanguageModel.echo
        let session = LanguageModelSession(model: model)

        let image = Transcript.ImageSegment(url: testImageURL)
        let response = try await session.respond(to: "Describe this image", image: image)

        #expect(response.content.contains("Describe this image"))

        // Verify transcript has prompt with text + image and a response
        #expect(session.transcript.count == 2)
        if case .prompt(let promptEntry) = session.transcript[0] {
            #expect(promptEntry.segments.count == 2)
            // Expect one text and one image segment
            let kinds = promptEntry.segments.map { segment -> String in
                switch segment {
                case .text: return "text";
                case .image: return "image";
                case .structure: return "structure"
                }
            }
            #expect(kinds.contains("text"))
            #expect(kinds.contains("image"))
        } else {
            Issue.record("First entry should be prompt with image")
        }
    }

    @Test func respondWithMultipleImages() async throws {
        let model = MockLanguageModel.echo
        let session = LanguageModelSession(model: model)

        let images: [Transcript.ImageSegment] = [
            .init(url: testImageURL),
            .init(data: testImageData, mimeType: "image/png"),
        ]
        let response = try await session.respond(to: "Classify these", images: images)

        #expect(response.content.contains("Classify these"))
        #expect(session.transcript.count == 2)
        if case .prompt(let promptEntry) = session.transcript[0] {
            #expect(promptEntry.segments.count == 3)
            let imageCount = promptEntry.segments.reduce(into: 0) { count, seg in if case .image = seg { count += 1 } }
            #expect(imageCount == 2)
        } else {
            Issue.record("First entry should be prompt with images")
        }
    }

    @Test func respondGeneratingStringWithImages() async throws {
        let model = MockLanguageModel.echo
        let session = LanguageModelSession(model: model)

        let images: [Transcript.ImageSegment] = [
            .init(url: testImageURL)
        ]
        let response: LanguageModelSession.Response<String> = try await session.respond(
            to: "What do you see?",
            images: images,
            generating: String.self,
            includeSchemaInPrompt: true
        )

        #expect(response.content.contains("What do you see?"))
        #expect(session.transcript.count == 2)
        if case .prompt(let promptEntry) = session.transcript[0] {
            #expect(promptEntry.segments.count == 2)
        } else {
            Issue.record("First entry should be prompt with image")
        }
    }

    @Test func streamResponseWithSingleImage() async throws {
        let model = MockLanguageModel.streamingMock()
        let session = LanguageModelSession(model: model)

        let image = Transcript.ImageSegment(url: testImageURL)
        let stream = session.streamResponse(to: "Stream about image", image: image)

        var snapshots = 0
        for try await _ in stream { snapshots += 1 }
        #expect(snapshots >= 1)

        // Prompt added at start, response added at end (append may occur after stream finishes)
        try await Task.sleep(for: .milliseconds(10))
        #expect(session.transcript.count == 2)
        if case .prompt(let promptEntry) = session.transcript[0] {
            #expect(promptEntry.segments.count == 2)
        } else {
            Issue.record("First entry should be prompt with image")
        }
    }

    @Test func streamResponseWithMultipleImages() async throws {
        let model = MockLanguageModel.streamingMock()
        let session = LanguageModelSession(model: model)

        let images: [Transcript.ImageSegment] = [
            .init(url: testImageURL),
            .init(data: testImageData, mimeType: "image/png"),
        ]
        let stream = session.streamResponse(to: "Stream about images", images: images)

        for try await _ in stream { /* drain */  }
        // Response append occurs after stream finishes; wait briefly
        try await Task.sleep(for: .milliseconds(10))
        #expect(session.transcript.count == 2)
        if case .prompt(let promptEntry) = session.transcript[0] {
            #expect(promptEntry.segments.count == 3)
            let imageCount = promptEntry.segments.reduce(into: 0) { c, seg in if case .image = seg { c += 1 } }
            #expect(imageCount == 2)
        } else {
            Issue.record("First entry should be prompt with images")
        }
    }
}
