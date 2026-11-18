import Testing

@testable import AnyLanguageModel

#if Llama
    @Suite("LlamaLanguageModel.SessionContextOwner")
    struct LlamaSessionContextOwnerTests {
        let model = LlamaLanguageModel(modelPath: "/path/to/model.gguf")

        @Test func ownerIsTheSessionThatItHolds() {
            let session = LanguageModelSession(model: model)
            let other = LanguageModelSession(model: model)
            let owner = LlamaLanguageModel.SessionContextOwner(session)
            #expect(owner.isOwner(session))
            #expect(!owner.isOwner(other))
        }

        /// A session made after the owner is freed isn't the owner,
        /// even when it gets the freed session's address,
        /// as the allocator usually gives it.
        @Test func sessionAtAFreedSessionsAddressIsNotTheOwner() {
            var owner: LlamaLanguageModel.SessionContextOwner?
            do {
                owner = .init(LanguageModelSession(model: model))
            }
            let sessions = (0 ..< 10).map { _ in LanguageModelSession(model: model) }
            for session in sessions {
                #expect(owner?.isOwner(session) == false)
            }
        }
    }
#endif
