import Foundation

final class Locked<State> {
    private let lock = NSLock()
    private var state: State

    init(_ state: State) {
        self.state = state
    }

    func access<T>(_ block: (inout State) throws -> T) rethrows -> T {
        try lock.withLock { try block(&self.state) }
    }
}

extension Locked: @unchecked Sendable where State: Sendable {}
