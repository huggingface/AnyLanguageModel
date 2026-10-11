import Foundation

protocol ProfileRequestIdentifiedLanguageModel {
    var profileRequestIdentity: UUID { get }
}

struct ProfileToolInvocation: Sendable {
    let call: Transcript.ToolCall
    let output: Transcript.ToolOutput
}

enum ProfileToolResolution: Sendable {
    case stop(calls: [Transcript.ToolCall])
    case invocations([ProfileToolInvocation])
}

extension LanguageModelSession {
    func profileContinuationResponse<Content: Generable>(
        ifSelectedModelDiffersFrom currentModelIdentity: UUID,
        requestContext: RequestContext,
        prompt: Prompt,
        generating type: Content.Type,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool,
        prefixEntries: [Transcript.Entry],
        prefixUsage: Usage,
        prefixText: String
    ) async throws -> Response<Content>? {
        if let selected = requestContext.model as? any ProfileRequestIdentifiedLanguageModel,
            selected.profileRequestIdentity == currentModelIdentity
        {
            return nil
        }

        let response: Response<Content>
        if type == GeneratedContent.self {
            let generated = try await requestContext.model.respond(
                within: self,
                to: prompt,
                schema: schema,
                includeSchemaInPrompt: requestContext.contextOptions.includeSchemaInPrompt
                    ?? includeSchemaInPrompt,
                options: requestContext.options
            )
            guard let typed = generated as? Response<Content> else {
                preconditionFailure("GeneratedContent response type mismatch")
            }
            response = typed
        } else {
            response = try await requestContext.model.respond(
                within: self,
                to: prompt,
                generating: type,
                includeSchemaInPrompt: requestContext.contextOptions.includeSchemaInPrompt
                    ?? includeSchemaInPrompt,
                options: requestContext.options
            )
        }

        var usage = prefixUsage
        usage.add(response.usage)
        var content = response.content
        var rawContent = response.rawContent
        if type == String.self, case .string(let suffix) = response.rawContent.kind {
            let combined = prefixText + suffix
            content = combined as! Content
            rawContent = GeneratedContent(combined)
        }
        return Response(
            content: content,
            rawContent: rawContent,
            transcriptEntries: ArraySlice(prefixEntries + response.transcriptEntries),
            usage: usage,
            providerMetadata: response.providerMetadata
        )
    }

    func profileContinuationStream<Content: Generable>(
        ifSelectedModelDiffersFrom currentModelIdentity: UUID,
        requestContext: RequestContext,
        prompt: Prompt,
        generating type: Content.Type,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool,
        prefixEntries: [Transcript.Entry],
        prefixUsage: Usage,
        prefixText: String
    ) -> ResponseStream<Content>? {
        if let selected = requestContext.model as? any ProfileRequestIdentifiedLanguageModel,
            selected.profileRequestIdentity == currentModelIdentity
        {
            return nil
        }

        let upstream: ResponseStream<Content>
        if type == GeneratedContent.self {
            let generated = requestContext.model.streamResponse(
                within: self,
                to: prompt,
                schema: schema,
                includeSchemaInPrompt: requestContext.contextOptions.includeSchemaInPrompt
                    ?? includeSchemaInPrompt,
                options: requestContext.options
            )
            guard let typed = generated as? ResponseStream<Content> else {
                preconditionFailure("GeneratedContent response stream type mismatch")
            }
            upstream = typed
        } else {
            upstream = requestContext.model.streamResponse(
                within: self,
                to: prompt,
                generating: type,
                includeSchemaInPrompt: requestContext.contextOptions.includeSchemaInPrompt
                    ?? includeSchemaInPrompt,
                options: requestContext.options
            )
        }

        let stream = AsyncThrowingStream<ResponseStream<Content>.Snapshot, any Error> { continuation in
            let task = Task {
                do {
                    for try await snapshot in upstream {
                        var usage = prefixUsage
                        usage.add(snapshot.usage)
                        var content = snapshot.content
                        var rawContent = snapshot.rawContent
                        if type == String.self, case .string(let suffix) = snapshot.rawContent.kind {
                            let combined = prefixText + suffix
                            content = combined as! Content.PartiallyGenerated
                            rawContent = GeneratedContent(combined)
                        }
                        continuation.yield(
                            .init(
                                content: content,
                                rawContent: rawContent,
                                transcriptEntries: ArraySlice(prefixEntries + snapshot.transcriptEntries),
                                usage: usage,
                                providerMetadata: snapshot.providerMetadata
                            )
                        )
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return ResponseStream(stream: stream)
    }

    func resolveProfileToolCalls(
        _ calls: [Transcript.ToolCall],
        requestContext: RequestContext,
        currentRoundEntries: [Transcript.Entry],
        missingToolOutput: @Sendable (Transcript.ToolCall) -> [Transcript.Segment] = { call in
            [.text(.init(content: "Tool not found: \(call.toolName)"))]
        }
    ) async throws -> ProfileToolResolution {
        guard !calls.isEmpty else { return .invocations([]) }

        if let delegate = toolExecutionDelegate {
            await delegate.didGenerateToolCalls(calls, in: self)
        }

        var decisions: [ToolExecutionDecision] = []
        decisions.reserveCapacity(calls.count)
        for call in calls {
            try await profileWillExecuteToolCall(
                call,
                requestContext: requestContext,
                currentRoundEntries: currentRoundEntries
            )
            let decision =
                await toolExecutionDelegate?.toolCallDecision(for: call, in: self) ?? .execute
            if case .stop = decision {
                return .stop(calls: calls)
            }
            decisions.append(decision)
        }

        var toolsByName: [String: any Tool] = [:]
        for tool in requestContext.tools where toolsByName[tool.name] == nil {
            toolsByName[tool.name] = tool
        }

        var invocations: [ProfileToolInvocation] = []
        invocations.reserveCapacity(calls.count)
        var completedEntries: [Transcript.Entry] = []
        for (call, decision) in zip(calls, decisions) {
            try Task.checkCancellation()
            let output: Transcript.ToolOutput
            switch decision {
            case .stop:
                return .stop(calls: calls)
            case .provideOutput(let segments):
                output = Transcript.ToolOutput(
                    id: call.id,
                    toolName: call.toolName,
                    segments: segments
                )
            case .execute:
                guard let tool = toolsByName[call.toolName] else {
                    output = Transcript.ToolOutput(
                        id: call.id,
                        toolName: call.toolName,
                        segments: missingToolOutput(call)
                    )
                    break
                }
                do {
                    let segments = try await withProfileToolExecution(
                        requestContext: requestContext,
                        currentRoundEntries: currentRoundEntries + completedEntries
                    ) {
                        try await tool.makeOutputSegments(from: call.arguments)
                    }
                    output = Transcript.ToolOutput(
                        id: call.id,
                        toolName: tool.name,
                        segments: segments
                    )
                } catch {
                    if let delegate = toolExecutionDelegate {
                        await delegate.didFailToolCall(call, error: error, in: self)
                    }
                    throw ToolCallError(tool: tool, underlyingError: error)
                }
            }

            invocations.append(.init(call: call, output: output))
            completedEntries.append(.toolOutput(output))
            try await profileDidProduceToolOutput(
                for: call,
                output: output,
                requestContext: requestContext,
                currentRoundEntries: currentRoundEntries + completedEntries
            )
            if let delegate = toolExecutionDelegate {
                await delegate.didExecuteToolCall(call, output: output, in: self)
            }
        }
        return .invocations(invocations)
    }
}
