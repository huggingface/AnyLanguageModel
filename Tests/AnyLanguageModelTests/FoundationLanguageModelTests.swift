import Foundation
import Testing
@testable import AnyLanguageModel

#if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS)
    import FoundationModels

    private let isFoundationLanguageModelAvailable = {
        if #available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *) {
            return FoundationModels.SystemLanguageModel.default.isAvailable
        }
        return false
    }()

    @Suite("FoundationLanguageModel")
    struct FoundationLanguageModelTests {
        @available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *)
        private actor Counter {
            private(set) var count = 0
            func increment() { count += 1 }
        }

        @Test(
            .enabled(if: ProcessInfo.processInfo.environment["CI"] == nil),
            .enabled(if: isFoundationLanguageModelAvailable),
            arguments: [false, true]
        )
        func dynamicSchemaOnDevice(_ streaming: Bool) async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *) else { return }
            let model = FoundationLanguageModel(FoundationModels.SystemLanguageModel.default)
            let session = AnyLanguageModel.LanguageModelSession(model: model)
            let schema = try SchemaResponseTests.schema()
            let prompt = "What is the capital of France? Put the city name in answer."
            let content: AnyLanguageModel.GeneratedContent
            if streaming {
                var last: AnyLanguageModel.GeneratedContent?
                for try await snapshot in session.streamResponse(to: prompt, schema: schema) {
                    last = snapshot.content
                }
                content = try #require(last)
            } else {
                content = try await session.respond(to: prompt, schema: schema).content
            }
            let answer = try SchemaResponseTests.Answer(content)
            #expect(answer.answer.contains("Paris"))
        }

        @Test func factoryRunsOnFirstLoadOnly() async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *) else { return }
            let counter = Counter()
            let model = FoundationLanguageModel {
                await counter.increment()
                return FoundationModels.PrivateCloudComputeLanguageModel()
            }
            #expect(await model.isLoaded == false)
            #expect(await counter.count == 0)
            try await model.load()
            try await model.load()
            #expect(await model.isLoaded == true)
            #expect(await counter.count == 1)
        }

        @Test func unloadReleasesTheModelAndReloadsOnDemand() async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *) else { return }
            let counter = Counter()
            let model = FoundationLanguageModel {
                await counter.increment()
                return FoundationModels.PrivateCloudComputeLanguageModel()
            }
            try await model.load()
            await model.unload()
            #expect(await model.isLoaded == false)
            #expect(await model.capabilities == nil)
            try await model.load()
            #expect(await counter.count == 2)
        }

        @Test func concurrentFirstRequestsShareOneFactoryRun() async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *) else { return }
            let counter = Counter()
            let model = FoundationLanguageModel {
                await counter.increment()
                try await Task.sleep(for: .milliseconds(50))
                return FoundationModels.PrivateCloudComputeLanguageModel()
            }
            async let first: Void = model.load()
            async let second: Void = model.load()
            async let third: Void = model.load()
            _ = try await (first, second, third)
            #expect(await model.isLoaded == true)
            #expect(await counter.count == 1)
        }

        @Test func concurrentLoadsPublishStateBeforeReturning() async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *) else { return }
            var incompleteLoads = 0
            for _ in 0 ..< 200 {
                let model = FoundationLanguageModel {
                    try await Task.sleep(for: .milliseconds(1))
                    return FoundationModels.PrivateCloudComputeLanguageModel()
                }
                incompleteLoads += try await withThrowingTaskGroup(of: Int.self) { group in
                    for _ in 0 ..< 20 {
                        group.addTask {
                            try await model.load()
                            // Check each caller before waiting for the rest of the group.
                            let isLoaded = await model.isLoaded
                            let capabilities = await model.capabilities
                            return isLoaded && capabilities != nil ? 0 : 1
                        }
                    }
                    var failures = 0
                    for try await value in group { failures += value }
                    return failures
                }
            }
            #expect(incompleteLoads == 0)
        }

        @Test func failedFactoryRunIsRetriedOnTheNextRequest() async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *) else { return }
            struct LoadFailure: Error {}
            let counter = Counter()
            let model = FoundationLanguageModel {
                await counter.increment()
                if await counter.count == 1 {
                    throw LoadFailure()
                }
                return FoundationModels.PrivateCloudComputeLanguageModel()
            }
            await #expect(throws: LoadFailure.self) {
                try await model.load()
            }
            #expect(await model.isLoaded == false)
            try await model.load()
            #expect(await model.isLoaded == true)
            #expect(await counter.count == 2)
        }

        @Test func unloadDuringLoadDiscardsTheResult() async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *) else { return }
            let counter = Counter()
            let model = FoundationLanguageModel {
                await counter.increment()
                try await Task.sleep(for: .milliseconds(50))
                return FoundationModels.PrivateCloudComputeLanguageModel()
            }
            let load = Task { try await model.load() }
            try await Task.sleep(for: .milliseconds(10))
            await model.unload()
            _ = try? await load.value
            #expect(await model.isLoaded == false)
            #expect(await counter.count == 1)
        }

        @Test func wrappingAnExistingModelIsLoadedImmediately() async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *) else { return }
            let model = FoundationLanguageModel(FoundationModels.PrivateCloudComputeLanguageModel())
            #expect(await model.isLoaded == true)
            #expect(model.isAvailable == true)
        }
    }
#endif
