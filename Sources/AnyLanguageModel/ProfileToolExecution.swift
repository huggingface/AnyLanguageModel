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
    /// The schema text for one request, under the setting that `requestContext` resolved.
    func schemaInstructions<Content: Generable>(
        for requestContext: RequestContext,
        generating type: Content.Type,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool
    ) -> String? {
        GenerationSchema.promptInstructions(
            generating: type,
            schema: schema,
            includeSchemaInPrompt: profileIncludeSchemaInPrompt(
                for: requestContext,
                callSiteValue: includeSchemaInPrompt
            )
        )
    }

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
        guard requestContext.hasDynamicProfile, !prefixEntries.isEmpty else { return nil }
        if let selected = requestContext.model as? any ProfileRequestIdentifiedLanguageModel,
            selected.profileRequestIdentity == currentModelIdentity
        {
            return nil
        }

        let includeSchemaInPrompt = profileIncludeSchemaInPrompt(
            for: requestContext,
            callSiteValue: includeSchemaInPrompt
        )
        let response: Response<Content>
        do {
            if type == GeneratedContent.self {
                let generated = try await requestContext.model.respond(
                    within: self,
                    to: prompt,
                    schema: schema,
                    includeSchemaInPrompt: includeSchemaInPrompt,
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
                    includeSchemaInPrompt: includeSchemaInPrompt,
                    options: requestContext.options
                )
            }
        } catch let stopped as ToolExecutionStopped {
            throw stopped.continuing(after: prefixEntries, usage: prefixUsage)
        }

        var earlierUsage = prefixUsage
        earlierUsage.add(response.requests.earlier)
        var content = response.content
        var rawContent = response.rawContent
        var providerMetadata = response.providerMetadata
        if type == String.self, case .string(let suffix) = response.rawContent.kind {
            let combined = prefixText + suffix
            content = combined as! Content
            rawContent = GeneratedContent(combined)
            providerMetadata = handoffResponseMetadata(
                providerMetadata,
                responseText: suffix,
                prefixText: prefixText,
                prefixEntries: prefixEntries
            )
        }
        return Response(
            content: content,
            rawContent: rawContent,
            transcriptEntries: ArraySlice(prefixEntries + response.transcriptEntries),
            requests: RequestsUsage(earlier: earlierUsage, final: response.requests.final),
            providerMetadata: providerMetadata
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
        guard requestContext.hasDynamicProfile, !prefixEntries.isEmpty else { return nil }
        if let selected = requestContext.model as? any ProfileRequestIdentifiedLanguageModel,
            selected.profileRequestIdentity == currentModelIdentity
        {
            return nil
        }

        let includeSchemaInPrompt = profileIncludeSchemaInPrompt(
            for: requestContext,
            callSiteValue: includeSchemaInPrompt
        )
        let upstream: ResponseStream<Content>
        if type == GeneratedContent.self {
            let generated = requestContext.model.streamResponse(
                within: self,
                to: prompt,
                schema: schema,
                includeSchemaInPrompt: includeSchemaInPrompt,
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
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: requestContext.options
            )
        }

        let stream = AsyncThrowingStream<ResponseStream<Content>.Snapshot, any Error> { continuation in
            let task = Task {
                do {
                    for try await snapshot in upstream {
                        var earlierUsage = prefixUsage
                        earlierUsage.add(snapshot.requests.earlier)
                        var unprepaidUsage = prefixUsage
                        unprepaidUsage.add(snapshot.unprepaidUsage)
                        var content = snapshot.content
                        var rawContent = snapshot.rawContent
                        var providerMetadata = snapshot.providerMetadata
                        if type == String.self, case .string(let suffix) = snapshot.rawContent.kind {
                            let combined = prefixText + suffix
                            content = combined as! Content.PartiallyGenerated
                            rawContent = GeneratedContent(combined)
                            providerMetadata = handoffResponseMetadata(
                                providerMetadata,
                                responseText: suffix,
                                prefixText: prefixText,
                                prefixEntries: prefixEntries
                            )
                        }
                        continuation.yield(
                            .init(
                                content: content,
                                rawContent: rawContent,
                                transcriptEntries: ArraySlice(prefixEntries + snapshot.transcriptEntries),
                                requests: RequestsUsage(earlier: earlierUsage, final: snapshot.requests.final),
                                providerMetadata: providerMetadata,
                                prepaid: snapshot.prepaid,
                                unprepaidUsage: unprepaidUsage
                            )
                        )
                    }
                    continuation.finish()
                } catch let stopped as ToolExecutionStopped {
                    continuation.finish(throwing: stopped.continuing(after: prefixEntries, usage: prefixUsage))
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return ResponseStream(stream: stream)
    }

    /// Resolves a round of tool calls the same way for every provider whose tool loop runs here.
    ///
    /// The delegate hears about the round, then decides each call in order; a `.stop` ends the
    /// round before any call runs. The calls then run at once, as Foundation Models runs them:
    /// a tool can be called concurrently with itself or with other tools, so one that isn't
    /// safe to run alongside others serializes its own work. Each sees the earlier rounds and
    /// every call in this round, but not the outputs of the other calls in this round.
    /// Once they're done, the profile and the delegate hear about each call in the order
    /// the model made them, one at a time, and the results keep that order.
    ///
    /// When a call throws, the round is cancelled first: the other calls are cancelled and
    /// waited for. Only then does the delegate hear about the error, and only the one the
    /// round throws.
    func resolveProfileToolCalls(
        _ calls: [Transcript.ToolCall],
        requestContext: RequestContext,
        currentRoundEntries: [Transcript.Entry],
        offeredTools: [any Tool]? = nil,
        missingToolOutput: @escaping @Sendable (Transcript.ToolCall) -> [Transcript.Segment] = { call in
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
        for tool in requestContext.toolsForExecution(in: self, offeredTools: offeredTools)
        where toolsByName[tool.name] == nil {
            toolsByName[tool.name] = tool
        }

        try Task.checkCancellation()
        let outputs: [Transcript.ToolOutput]
        do {
            outputs = try await withThrowingTaskGroup(of: (Int, Transcript.ToolOutput).self) { group in
                for (index, (call, decision)) in zip(calls, decisions).enumerated() {
                    let tool = toolsByName[call.toolName]
                    group.addTask {
                        let output = try await self.toolOutput(
                            for: call,
                            decision: decision,
                            tool: tool,
                            requestContext: requestContext,
                            currentRoundEntries: currentRoundEntries,
                            missingToolOutput: missingToolOutput
                        )
                        return (index, output)
                    }
                }
                var outputs = [Transcript.ToolOutput?](repeating: nil, count: calls.count)
                for try await (index, output) in group {
                    outputs[index] = output
                }
                return outputs.compactMap { $0 }
            }
        } catch let failure as ToolCallFailure {
            // The group has cancelled and waited for the other calls by now.
            if let delegate = toolExecutionDelegate {
                await delegate.didFailToolCall(failure.call, error: failure.underlyingError, in: self)
            }
            throw ToolCallError(tool: failure.tool, underlyingError: failure.underlyingError)
        }

        var invocations: [ProfileToolInvocation] = []
        invocations.reserveCapacity(calls.count)
        var completedEntries: [Transcript.Entry] = []
        for (call, output) in zip(calls, outputs) {
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

    /// One call's output: the delegate's, a missing tool's, or the tool's own.
    private func toolOutput(
        for call: Transcript.ToolCall,
        decision: ToolExecutionDecision,
        tool: (any Tool)?,
        requestContext: RequestContext,
        currentRoundEntries: [Transcript.Entry],
        missingToolOutput: @Sendable (Transcript.ToolCall) -> [Transcript.Segment]
    ) async throws -> Transcript.ToolOutput {
        switch decision {
        case .provideOutput(let segments):
            return Transcript.ToolOutput(id: call.id, toolName: call.toolName, segments: segments)
        case .stop, .execute:
            // A `.stop` ends the round before any call runs, so here it's `.execute`.
            guard let tool else {
                return Transcript.ToolOutput(id: call.id, toolName: call.toolName, segments: missingToolOutput(call))
            }
            do {
                let segments = try await withProfileToolExecution(
                    requestContext: requestContext,
                    currentRoundEntries: currentRoundEntries
                ) {
                    try await tool.makeOutputSegments(from: call.arguments)
                }
                return Transcript.ToolOutput(id: call.id, toolName: tool.name, segments: segments)
            } catch {
                throw ToolCallFailure(call: call, tool: tool, underlyingError: error)
            }
        }
    }
}

/// A tool call that threw, kept until the round has cancelled the others.
private struct ToolCallFailure: Error {
    let call: Transcript.ToolCall
    let tool: any Tool
    let underlyingError: any Error
}
