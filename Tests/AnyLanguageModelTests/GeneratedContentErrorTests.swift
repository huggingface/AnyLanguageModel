import Testing

import AnyLanguageModel

@Suite("Generated content errors")
struct GeneratedContentErrorTests {
    @Test func conversionErrorAliasPreservesEqualityAndHashing() {
        let error = GeneratedContentConversionError.typeMismatch
        #expect(error == .typeMismatch)
        #expect(error != .neverCannotBeInstantiated)

        let errors: Set<GeneratedContentConversionError> = [
            .typeMismatch, .typeMismatch, .neverCannotBeInstantiated,
        ]
        #expect(errors.count == 2)
        #expect(errors.contains(error))
        #expect(errors.contains(.neverCannotBeInstantiated))
    }
}
