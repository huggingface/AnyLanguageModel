import Testing

@testable import AnyLanguageModel

@Generable
private struct ContradictingGuides {
    @Guide(description: "A count", .minimum(10), .maximum(5))
    var count: Int
}

@Suite("Number bounds")
struct BoundsTests {
    @Test func boundsAreMadeOnlyInOrder() {
        #expect(Bounds<Int>(minimum: nil, maximum: nil) == .unbounded)
        #expect(Bounds(minimum: 1, maximum: nil) == .from(1))
        #expect(Bounds(minimum: nil, maximum: 9) == .through(9))
        #expect(Bounds(minimum: 1, maximum: 9) == .closed(1 ... 9))
        #expect(Bounds(minimum: 9, maximum: 1) == nil)
    }

    @Test func containsAndClamped() {
        let bounds = Bounds.closed(2 ... 4)
        #expect(bounds.contains(3))
        #expect(!bounds.contains(5))
        #expect(bounds.clamped(0) == 2)
        #expect(bounds.clamped(3) == 3)
        #expect(bounds.clamped(9) == 4)
        #expect(Bounds.from(5).clamped(0) == 5)
        #expect(Bounds.through(-3).clamped(0) == -3)
        #expect(!Bounds.from(5).contains(4))
    }

    /// An integer guide keeps its bounds as integers, beyond what a `Double` holds exactly.
    @Test func integerGuideBoundsStayIntegers() {
        let guide = GenerationGuide<Int>.range(Int.max - 1 ... Int.max)
        #expect(guide.minimum == .integer(Int.max - 1))
        #expect(guide.maximum == .integer(Int.max))
    }

    #if os(macOS) || os(Linux)
        /// Guides that no value satisfies are a mistake in the code, as an inverted `ClosedRange` is.
        @Test func contradictingGuidesStop() async {
            await #expect(processExitsWith: .failure) {
                _ = ContradictingGuides.generationSchema
            }
        }
    #endif
}
