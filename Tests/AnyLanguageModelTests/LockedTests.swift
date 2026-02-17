import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("Locked Tests")
struct LockedTests {
    @Test("Read access returns the initial value")
    func readAccess() {
        let locked = Locked(42)
        let value = locked.withLock { $0 }
        #expect(value == 42)
    }

    @Test("Write access mutates the state")
    func writeAccess() {
        let locked = Locked(0)
        locked.withLock { $0 = 99 }
        let value = locked.withLock { $0 }
        #expect(value == 99)
    }

    @Test("Access returns the value from the closure")
    func returnValue() {
        let locked = Locked("hello")
        let result = locked.withLock { state -> Int in
            state += " world"
            return state.count
        }
        #expect(result == 11)
        #expect(locked.withLock { $0 } == "hello world")
    }

    @Test("Access propagates thrown errors")
    func throwsPropagation() {
        struct TestError: Error, Equatable {}

        let locked = Locked(0)
        #expect(throws: TestError.self) {
            try locked.withLock { _ in throw TestError() }
        }
    }

    @Test("Works with complex state types")
    func complexState() {
        struct State {
            var name: String
            var count: Int
            var tags: [String]
        }

        let locked = Locked(State(name: "initial", count: 0, tags: []))
        locked.withLock { state in
            state.name = "updated"
            state.count = 5
            state.tags.append("a")
            state.tags.append("b")
        }

        let snapshot = locked.withLock { $0 }
        #expect(snapshot.name == "updated")
        #expect(snapshot.count == 5)
        #expect(snapshot.tags == ["a", "b"])
    }

    @Test("Concurrent increments produce the correct final count")
    func concurrentCounter() async {
        let locked = Locked(0)
        let iterations = 100_000

        await withTaskGroup(of: Void.self) { group in
            for _ in 0 ..< iterations {
                group.addTask { locked.withLock { $0 += 1 } }
            }
        }

        let finalValue = locked.withLock { $0 }
        #expect(finalValue == iterations)
    }

    @Test("Concurrent increments with mixed task priorities")
    func concurrentCounterWithMixedPriorities() async {
        let locked = Locked(0)
        let iterations = 100_000

        await withTaskGroup(of: Void.self) { group in
            for i in 0 ..< iterations {
                let priority: TaskPriority = i.isMultiple(of: 2) ? .high : .background
                group.addTask(priority: priority) {
                    locked.withLock { $0 += 1 }
                }
            }
        }

        let finalValue = locked.withLock { $0 }
        #expect(finalValue == iterations)
    }

    @Test("Concurrent appends to an array")
    func concurrentArrayAppend() async {
        let locked = Locked<[Int]>([])
        let iterations = 10_000

        await withTaskGroup(of: Void.self) { group in
            for i in 0 ..< iterations {
                group.addTask { locked.withLock { $0.append(i) } }
            }
        }

        let finalArray = locked.withLock { $0 }
        #expect(finalArray.count == iterations)
    }

    @Test("Multiple independent locked values mutated concurrently")
    func multipleLockedValues() async {
        let lockedA = Locked(0)
        let lockedB = Locked(0)
        let iterations = 50_000

        await withTaskGroup(of: Void.self) { group in
            for _ in 0 ..< iterations {
                group.addTask { lockedA.withLock { $0 += 1 } }
                group.addTask { lockedB.withLock { $0 += 1 } }
            }
        }

        #expect(lockedA.withLock { $0 } == iterations)
        #expect(lockedB.withLock { $0 } == iterations)
    }

    @Test("Can wrap a non-Sendable type")
    func nonSendableElement() {
        class Box {
            var value: Int
            init(_ value: Int) { self.value = value }
        }

        let locked = Locked(Box(10))
        locked.withLock { $0.value += 5 }
        let result = locked.withLock { $0.value }
        #expect(result == 15)
    }

    @Test("Copies share the same underlying storage")
    func copySharesStorage() {
        let original = Locked(0)
        let copy = original
        original.withLock { $0 = 42 }
        let value = copy.withLock { $0 }
        #expect(value == 42)
    }
}
