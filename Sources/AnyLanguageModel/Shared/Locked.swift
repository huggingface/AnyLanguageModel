import Foundation

/// Protects shared mutable state behind an `NSLock`.
final class Locked<State> {
    private let lock = NSLock()
    private var state: State

    /// Creates a locked container with the given initial state.
    init(_ state: State) {
        self.state = state
    }

    /// Executes `body` while holding the lock.
    ///
    /// - Parameter body: A closure that reads or mutates the protected state.
    /// - Returns: The value returned by `body`.
    /// - Throws: Rethrows any error from `body`.
    /// - Note: Keep critical sections small and synchronous.
    func withLock<T>(_ body: (inout State) throws -> T) rethrows -> T {
        try lock.withLock { try body(&self.state) }
    }
}

extension Locked: @unchecked Sendable where State: Sendable {}
