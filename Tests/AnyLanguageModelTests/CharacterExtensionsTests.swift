import Testing

@testable import AnyLanguageModel

@Suite("Character Extensions")
struct CharacterExtensionsTests {
    private func character(_ string: String) -> Character {
        string.first!
    }

    @Test func containsEmojiScalarDetectsEmoji() {
        #expect(character("😀").containsEmojiScalar)
        #expect(!character("A").containsEmojiScalar)
    }

    @Test func isValidJSONStringCharacterAcceptsExpectedCharacters() {
        #expect(character("A").isValidJSONStringCharacter)
        #expect(character("7").isValidJSONStringCharacter)
        #expect(character(" ").isValidJSONStringCharacter)
        #expect(character("😀").isValidJSONStringCharacter)
        #expect(character("é").isValidJSONStringCharacter)
    }

    @Test func isValidJSONStringCharacterRejectsDisallowedCharacters() {
        let control = Character(UnicodeScalar(0x1F)!)

        #expect(!character("\\").isValidJSONStringCharacter)
        #expect(!character("\"").isValidJSONStringCharacter)
        #expect(!character("”").isValidJSONStringCharacter)
        #expect(!control.isValidJSONStringCharacter)
        #expect(!character("】").isValidJSONStringCharacter)
    }
}
