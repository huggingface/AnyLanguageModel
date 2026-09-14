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

        @Test func wrappingAnExistingModelIsLoadedImmediately() async throws {
            guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, watchOS 27.0, *) else { return }
            let model = FoundationLanguageModel(FoundationModels.PrivateCloudComputeLanguageModel())
            #expect(await model.isLoaded == true)
            #expect(model.isAvailable == true)
        }
    }
#endif
