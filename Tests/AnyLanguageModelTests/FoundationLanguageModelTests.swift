import Testing
@testable import AnyLanguageModel

#if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS)
    import FoundationModels

    @Suite("FoundationLanguageModel")
    struct FoundationLanguageModelTests {
        @available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *)
        private actor Counter {
            private(set) var count = 0
            func increment() { count += 1 }
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
