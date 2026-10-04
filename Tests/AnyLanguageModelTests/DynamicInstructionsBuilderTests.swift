import Testing

@testable import AnyLanguageModel

@Suite("Dynamic instructions builder")
struct DynamicInstructionsBuilderTests {
    @Test func builderComposesNestedConditionalEmptyAndToolArrayContent() {
        let resolved = AnyDynamicInstructions(erasing: Composition(enabled: true)).resolveForRequest()

        #expect(resolved.instructions?.description == "Outer\nNested\nFor each")
        #expect(resolved.tools.map(\.name) == ["tool-a", "tool-b"])
    }

    @Test func falseConditionLeavesOutItsContent() {
        let resolved = AnyDynamicInstructions(erasing: Composition(enabled: false)).resolveForRequest()

        #expect(resolved.instructions?.description == "Outer\nFor each")
        #expect(resolved.tools.isEmpty)
    }

    @Test func bodyIsEvaluatedOnEachResolution() {
        let counter = Counter()
        let dynamic = AnyDynamicInstructions(erasing: CountingInstructions(counter: counter))

        #expect(dynamic.resolveForRequest().instructions?.description == "Request 1")
        #expect(dynamic.resolveForRequest().instructions?.description == "Request 2")
    }

    @Test func nestingDoesNotChangeWhitespace() {
        let flat = AnyDynamicInstructions(erasing: Flat()).resolveForRequest()
        let nested = AnyDynamicInstructions(erasing: Outer()).resolveForRequest()

        #expect(flat.instructions?.description == "A\nB  \nC")
        #expect(nested.instructions?.description == flat.instructions?.description)
    }

    @Test func emptyBuilderResolvesToNothing() {
        let resolved = AnyDynamicInstructions(erasing: EmptyDynamicInstructions()).resolveForRequest()

        #expect(resolved.instructions == nil)
        #expect(resolved.tools.isEmpty)
    }
}

private struct Composition: DynamicInstructions {
    let enabled: Bool

    var body: some DynamicInstructions {
        Instructions("Outer")
        if enabled {
            Nested()
        }
        EmptyDynamicInstructions()
        ForEach([Item(id: 1, text: "For each")]) { item in
            Instructions(item.text)
        }
    }
}

private struct Flat: DynamicInstructions {
    var body: some DynamicInstructions {
        Instructions("A")
        Instructions("B  ")
        Instructions("C")
    }
}

private struct Outer: DynamicInstructions {
    var body: some DynamicInstructions {
        Instructions("A")
        Inner()
    }
}

private struct Inner: DynamicInstructions {
    var body: some DynamicInstructions {
        Instructions("B  ")
        Instructions("C")
    }
}

private struct Item: Identifiable {
    let id: Int
    let text: String
}

private struct Nested: DynamicInstructions {
    var body: some DynamicInstructions {
        Instructions("Nested")
        [NamedTool(name: "tool-a"), NamedTool(name: "tool-b")] as [any Tool]
    }
}

private struct NamedTool: Tool {
    let name: String
    let description = "A tool that returns its name"

    typealias Arguments = GeneratedContent

    var parameters: GenerationSchema {
        GeneratedContent.generationSchema
    }

    func call(arguments: GeneratedContent) async throws -> String {
        name
    }
}

private final class Counter: @unchecked Sendable {
    private let count = Locked(0)

    func next() -> Int {
        count.withLock { value in
            value += 1
            return value
        }
    }
}

private struct CountingInstructions: DynamicInstructions {
    let counter: Counter

    var body: some DynamicInstructions {
        Instructions("Request \(counter.next())")
    }
}
