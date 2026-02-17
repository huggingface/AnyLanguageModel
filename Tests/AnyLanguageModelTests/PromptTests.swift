import Testing

@testable import AnyLanguageModel

@Suite("Prompt")
struct PromptTests {
    @Test func initializesFromStringRepresentable() {
        let prompt = Prompt("Hello")
        #expect(prompt.description == "Hello")
    }

    @Test func initializesFromExistingPromptRepresentable() {
        let existing = Prompt("Existing")
        let wrapped = Prompt(existing)
        #expect(wrapped.description == "Existing")
    }

    @Test func initializesFromArrayRepresentable() {
        let prompt = Prompt(["One", "Two", "Three"])
        #expect(prompt.description == "One\nTwo\nThree")
    }

    @Test func builderSupportsConditionalsAndOptionals() {
        let includeExtra = true
        let maybeLine: String? = nil

        let prompt = Prompt {
            "Always"
            if includeExtra {
                "Conditional"
            } else {
                "Other"
            }
            if let maybeLine {
                maybeLine
            }
        }

        #expect(prompt.description == "Always\nConditional")
    }

    @Test func builderTrimsLeadingAndTrailingWhitespaceAndNewlines() {
        let prompt = Prompt {
            "\n  First line"
            "Second line\n"
        }

        #expect(prompt.description == "First line\nSecond line")
    }
}
