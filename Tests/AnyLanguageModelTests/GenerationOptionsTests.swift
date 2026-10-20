import Testing

@testable import AnyLanguageModel

@Suite("Generation options")
struct GenerationOptionsTests {
    @Test func maximumToolRoundsDefaultsToEight() {
        #expect(GenerationOptions().maximumToolRounds == 8)
    }

    #if os(macOS) || os(Linux)
        /// With no round allowed, every response that calls a tool would fail at once, so it's refused.
        @Test func maximumToolRoundsBelowOneStops() async {
            await #expect(processExitsWith: .failure) {
                var options = GenerationOptions()
                options.maximumToolRounds = 0
            }
        }
    #endif
}
