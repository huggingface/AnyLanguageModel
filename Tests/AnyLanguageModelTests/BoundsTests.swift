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

    @Test func replacingABoundDropsTheOneItContradicts() {
        #expect(Bounds.closed(1 ... 9).replacing(minimum: 5) == .closed(5 ... 9))
        #expect(Bounds.closed(1 ... 9).replacing(minimum: 10) == .from(10))
        #expect(Bounds.closed(1 ... 9).replacing(maximum: 5) == .closed(1 ... 5))
        #expect(Bounds.closed(1 ... 9).replacing(maximum: 0) == .through(0))
        #expect(Bounds<Int>.unbounded.replacing(minimum: 3) == .from(3))
    }

    /// Guides that no value satisfies together still make a schema:
    /// the later guide's bound replaces the earlier one it contradicts.
    @Test func contradictingGuidesKeepTheLaterBound() throws {
        func bounds(of property: GenerationSchema.Property) -> GenerationSchema.NumberNode? {
            if case .number(let node) = property.node { node } else { nil }
        }
        let count = GenerationSchema.Property(name: "count", type: Int.self, guides: [.minimum(10), .maximum(5)])
        #expect(bounds(of: count) == .integer(description: nil, bounds: .through(5)))
        let level = GenerationSchema.Property(name: "level", type: Int.self, guides: [.maximum(5), .minimum(10)])
        #expect(bounds(of: level) == .integer(description: nil, bounds: .from(10)))
        let share = GenerationSchema.Property(name: "share", type: Double.self, guides: [.minimum(1), .maximum(0.5)])
        #expect(bounds(of: share) == .floatingPoint(description: nil, bounds: .through(0.5)))
    }
}
