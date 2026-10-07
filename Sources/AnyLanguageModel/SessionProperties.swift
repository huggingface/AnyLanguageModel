import Foundation
import Observation

extension Transcript {
    /// The mutable, instruction-free history exposed through session properties.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel on OS 26. It mirrors
    ///   Foundation Models 27's `Transcript.HistoryView` role with value semantics.
    public typealias HistoryView = [Entry]
}

/// A type-keyed session property.
///
/// - Note: This API is exclusive to AnyLanguageModel on OS 26 and mirrors the
///   Foundation Models 27 session-property contract.
public protocol SessionPropertyKey: SendableMetatype {
    associatedtype Value
    static var defaultValue: Value { get }
}

/// Declares a stored value on ``SessionPropertyValues``.
@attached(accessor)
@attached(peer, names: prefixed(__Key_))
public macro SessionPropertyEntry() =
    #externalMacro(module: "AnyLanguageModelMacros", type: "SessionPropertyEntryMacro")

/// Session-scoped storage shared by profiles, dynamic instructions, and tools.
///
/// - Note: This API is exclusive to AnyLanguageModel on OS 26 and mirrors
///   Foundation Models 27.
@Observable
public final class SessionPropertyValues: @unchecked Sendable {
    @ObservationIgnored private let customValues = Locked<[ObjectIdentifier: Any]>([:])
    @ObservationIgnored private let historyGetter: @Sendable () -> Transcript.HistoryView
    @ObservationIgnored private let historySetter: @Sendable (Transcript.HistoryView) -> Void

    init(
        historyGetter: @escaping @Sendable () -> Transcript.HistoryView,
        historySetter: @escaping @Sendable (Transcript.HistoryView) -> Void
    ) {
        self.historyGetter = historyGetter
        self.historySetter = historySetter
    }

    /// The canonical session history after dynamic instructions.
    public var history: Transcript.HistoryView {
        get { SessionPropertyBinding.history?.entries ?? historyGetter() }
        set {
            if let history = SessionPropertyBinding.history {
                precondition(
                    history.isWritable,
                    "Session history is read-only while dynamic instructions or a Tool is active"
                )
                history.entries = newValue
                history.persist(newValue)
                return
            }
            historySetter(newValue)
        }
    }

    public subscript<Key>(key: Key.Type) -> Key.Value where Key: SessionPropertyKey {
        get {
            customValues.withLock { values in
                if let value = values[ObjectIdentifier(key)] as? Key.Value { return value }
                let value = Key.defaultValue
                values[ObjectIdentifier(key)] = value
                return value
            }
        }
        set {
            customValues.withLock { values in
                values[ObjectIdentifier(key)] = newValue
            }
        }
    }

    func historyBinding(
        _ entries: Transcript.HistoryView,
        isWritable: Bool,
        protecting protectedEntryIDs: Set<String> = []
    ) -> SessionHistoryBinding {
        let persist: (@Sendable (Transcript.HistoryView) -> Void)?
        if isWritable {
            persist = { [historySetter] updated in
                historySetter(
                    updated.filter { !protectedEntryIDs.contains($0.id) }
                )
            }
        } else {
            persist = nil
        }
        return SessionHistoryBinding(
            entries,
            isWritable: isWritable,
            persist: persist
        )
    }
}

enum SessionPropertyBinding {
    @TaskLocal static var values: SessionPropertyValues?
    @TaskLocal static var history: SessionHistoryBinding?
}

final class SessionHistoryBinding: @unchecked Sendable {
    private let storage: Locked<Transcript.HistoryView>
    let isWritable: Bool
    private let persistValue: (@Sendable (Transcript.HistoryView) -> Void)?

    init(
        _ entries: Transcript.HistoryView,
        isWritable: Bool,
        persist: (@Sendable (Transcript.HistoryView) -> Void)? = nil
    ) {
        storage = Locked(entries)
        self.isWritable = isWritable
        self.persistValue = persist
    }

    var entries: Transcript.HistoryView {
        get { storage.withLock { $0 } }
        set { storage.withLock { $0 = newValue } }
    }

    func persist(_ entries: Transcript.HistoryView) {
        persistValue?(entries)
    }
}

extension LanguageModelSession {
    /// Accesses a value in the currently evaluating session.
    @propertyWrapper
    public struct SessionProperty<Value> {
        private let keyPath: ReferenceWritableKeyPath<SessionPropertyValues, Value>

        public init(_ keyPath: ReferenceWritableKeyPath<SessionPropertyValues, Value>) {
            self.keyPath = keyPath
        }

        public var wrappedValue: Value {
            get {
                guard let values = SessionPropertyBinding.values else {
                    preconditionFailure(
                        "SessionProperty can only be accessed while a session profile or Tool is active"
                    )
                }
                return values[keyPath: keyPath]
            }
            nonmutating set {
                guard let values = SessionPropertyBinding.values else {
                    preconditionFailure(
                        "SessionProperty can only be accessed while a session profile or Tool is active"
                    )
                }
                values[keyPath: keyPath] = newValue
            }
        }
    }
}

extension LanguageModelSession.SessionProperty: @unchecked Sendable where Value: Sendable {}

extension DynamicInstructions {
    public typealias SessionProperty = LanguageModelSession.SessionProperty
}

extension Tool {
    public typealias SessionProperty = LanguageModelSession.SessionProperty
}
