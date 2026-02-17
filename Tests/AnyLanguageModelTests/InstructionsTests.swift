import Testing

@testable import AnyLanguageModel

@Suite("Instructions")
struct InstructionsTests {
    @Test func initializesFromStringRepresentable() {
        let instructions = Instructions("Be concise.")
        #expect(instructions.description == "Be concise.")
    }

    @Test func initializesFromInstructionsRepresentable() {
        let existing = Instructions("Base instructions")
        let wrapped = Instructions(existing)
        #expect(wrapped.description == "Base instructions")
    }

    @Test func builderCombinesLines() throws {
        let instructions = try Instructions {
            "First line"
            "Second line"
        }

        #expect(instructions.description == "First line\nSecond line")
    }

    @Test func builderSupportsConditionalsAndOptionals() throws {
        let includeConditional = true
        let includeOptional = false

        let instructions = try Instructions {
            "Always"
            if includeConditional {
                "Conditional"
            } else {
                "Other"
            }
            if includeOptional {
                "Optional"
            }
        }

        #expect(instructions.description == "Always\nConditional")
    }

    @Test func arrayRepresentationJoinsByNewline() {
        let array = ["One", "Two", "Three"]
        #expect(array.instructionsRepresentation.description == "One\nTwo\nThree")
    }
}
