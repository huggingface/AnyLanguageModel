import Foundation
import Observation

extension Transcript {
    /// The mutable, instruction-free history exposed through session properties.
    ///
    /// - Note: This API is exclusive to AnyLanguageModel on OS 26. It mirrors
    ///   Foundation Models 27's `Transcript.HistoryView` role with value semantics.
    public struct HistoryView: RandomAccessCollection, MutableCollection, RangeReplaceableCollection,
        ExpressibleByArrayLiteral, Sendable
    {
        public typealias Element = Entry
        public typealias SubSequence = HistoryView
        public typealias Indices = Range<Index>
        private var entries: ArraySlice<Entry>

        public struct Index: Comparable, Hashable, Strideable, Sendable {
            fileprivate let offset: Int

            public static func < (lhs: Self, rhs: Self) -> Bool { lhs.offset < rhs.offset }
            public func distance(to other: Self) -> Int { other.offset - offset }
            public func advanced(by n: Int) -> Self { Self(offset: offset + n) }
        }

        public init() { entries = [] }
        public init<S: Sequence>(_ elements: S) where S.Element == Entry { entries = Array(elements)[...] }
        public init(arrayLiteral elements: Entry...) { entries = elements[...] }
        private init(entries: ArraySlice<Entry>) { self.entries = entries }

        public var startIndex: Index { Index(offset: entries.startIndex) }
        public var endIndex: Index { Index(offset: entries.endIndex) }
        public var indices: Indices { startIndex ..< endIndex }
        public func index(after i: Index) -> Index { i.advanced(by: 1) }
        public func index(before i: Index) -> Index { i.advanced(by: -1) }
        public func index(_ i: Index, offsetBy distance: Int) -> Index { i.advanced(by: distance) }
        public func distance(from start: Index, to end: Index) -> Int { start.distance(to: end) }

        public subscript(position: Index) -> Entry {
            get { entries[position.offset] }
            set { entries[position.offset] = newValue }
        }

        public subscript(bounds: Range<Index>) -> HistoryView {
            get { HistoryView(entries: entries[bounds.lowerBound.offset ..< bounds.upperBound.offset]) }
            set { replaceSubrange(bounds, with: newValue) }
        }

        public mutating func replaceSubrange<C: Collection>(_ subrange: Range<Index>, with newElements: C)
        where C.Element == Entry {
            entries.replaceSubrange(subrange.lowerBound.offset ..< subrange.upperBound.offset, with: newElements)
        }
    }
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

/// Session-scoped storage shared by dynamic instructions and Tools.
///
/// - Note: This API is exclusive to AnyLanguageModel on OS 26 and mirrors
///   Foundation Models 27.
@Observable
public final class SessionPropertyValues: @unchecked Sendable {
    @ObservationIgnored private let sessionPropertyRegistrar = ObservationRegistrar()
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

    /// The session history without an instructions entry.
    public var history: Transcript.HistoryView {
        get {
            _accessSessionProperty(keyPath: \.history)
            if SessionPropertyBinding.values === self, let history = SessionPropertyBinding.history {
                return history
            }
            return historyGetter()
        }
        set {
            _withSessionPropertyMutation(keyPath: \.history) {
                precondition(
                    SessionPropertyBinding.values !== self || SessionPropertyBinding.history == nil,
                    "Session history is read-only while dynamic instructions or a Tool is active"
                )
                historySetter(newValue)
            }
        }
    }

    public subscript<Key>(key: Key.Type) -> Key.Value where Key: SessionPropertyKey {
        get {
            if let value = customValues.withLock({ $0[ObjectIdentifier(key)] as? Key.Value }) {
                return value
            }
            let defaultValue = Key.defaultValue
            return customValues.withLock { values in
                if let value = values[ObjectIdentifier(key)] as? Key.Value { return value }
                values[ObjectIdentifier(key)] = defaultValue
                return defaultValue
            }
        }
        set {
            customValues.withLock { values in
                values[ObjectIdentifier(key)] = newValue
            }
        }
    }

    @_documentation(visibility: internal)
    public func _accessSessionProperty<Member>(
        keyPath: KeyPath<SessionPropertyValues, Member>
    ) {
        sessionPropertyRegistrar.access(self, keyPath: keyPath)
    }

    @_documentation(visibility: internal)
    public func _withSessionPropertyMutation<Member, Result>(
        keyPath: KeyPath<SessionPropertyValues, Member>,
        _ mutation: () throws -> Result
    ) rethrows -> Result {
        try sessionPropertyRegistrar.withMutation(
            of: self,
            keyPath: keyPath,
            mutation
        )
    }
}

enum SessionPropertyBinding {
    @TaskLocal static var values: SessionPropertyValues?
    @TaskLocal static var history: Transcript.HistoryView?
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
                        "SessionProperty can only be accessed while dynamic instructions or a Tool is active"
                    )
                }
                return values[keyPath: keyPath]
            }
            nonmutating set {
                guard let values = SessionPropertyBinding.values else {
                    preconditionFailure(
                        "SessionProperty can only be accessed while dynamic instructions or a Tool is active"
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

    func bindingSessionProperties(_ properties: SessionPropertyValues, history: Transcript.HistoryView) -> any Tool {
        SessionPropertyTool(wrapped: self, properties: properties, history: history)
    }
}

private struct SessionPropertyTool<Wrapped: Tool>: Tool {
    let wrapped: Wrapped
    let properties: SessionPropertyValues
    let history: Transcript.HistoryView

    var name: String { wrapped.name }
    var description: String { wrapped.description }
    var parameters: GenerationSchema { wrapped.parameters }
    var includesSchemaInInstructions: Bool { wrapped.includesSchemaInInstructions }

    func call(arguments: Wrapped.Arguments) async throws -> Wrapped.Output {
        let history =
            SessionPropertyBinding.values === properties
            ? SessionPropertyBinding.history ?? history : history
        return try await SessionPropertyBinding.$values.withValue(properties) {
            try await SessionPropertyBinding.$history.withValue(history) {
                try await wrapped.call(arguments: arguments)
            }
        }
    }
}
