import Testing

@testable import AnyLanguageModel

@Generable
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
        /// Given apart, they reach the schema's own check, which says so.
        @Test func contradictingIntegerGuidesStop() async throws {
            let result = try await #require(processExitsWith: .failure, observing: [\.standardErrorContent]) {
                _ = GenerationSchema.Property(name: "count", type: Int.self, guides: [.minimum(10), .maximum(5)])
            }
            #expect(
                String(decoding: result.standardErrorContent, as: UTF8.self).contains(
                    "minimum 10 is above its maximum 5"
                )
            )
        }

        @Test func contradictingFloatingPointGuidesStop() async throws {
            let result = try await #require(processExitsWith: .failure, observing: [\.standardErrorContent]) {
                _ = GenerationSchema.Property(name: "share", type: Double.self, guides: [.minimum(1), .maximum(0.5)])
            }
            #expect(
                String(decoding: result.standardErrorContent, as: UTF8.self).contains(
                    "minimum 1.0 is above its maximum 0.5"
                )
            )
        }
    #endif
}
