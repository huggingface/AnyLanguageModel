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

/// Makes a session property key's default value a computed property,
/// so that each session gets its own value.
///
/// ``SessionPropertyEntry()`` applies this macro; don't use it directly.
@attached(accessor)
public macro __SessionPropertyEntryDefaultValue() =
    #externalMacro(module: "AnyLanguageModelMacros", type: "SessionPropertyEntryDefaultValueMacro")

/// Session-scoped storage shared by profiles, dynamic instructions, and tools.
///
/// - Note: This API is exclusive to AnyLanguageModel on OS 26 and mirrors
///   Foundation Models 27.
@Observable
public final class SessionPropertyValues: @unchecked Sendable {
    @ObservationIgnored private let sessionPropertyRegistrar = ObservationRegistrar()
    @ObservationIgnored private let customValues = Locked<[ObjectIdentifier: Any]>([:])
    @ObservationIgnored private let historyGetter: @Sendable () -> [Transcript.Entry]
    @ObservationIgnored private let historySetter: @Sendable ([Transcript.Entry]) -> Void
    @ObservationIgnored private let historyReplacer: @Sendable ([Transcript.Entry], UInt64) -> UInt64?

    /// Creates session properties backed by a session's history.
    ///
    /// `historyReplacer` replaces the history only if it's still at the given version,
    /// and returns the new version, or `nil` if the history changed.
    init(
        historyGetter: @escaping @Sendable () -> [Transcript.Entry],
        historySetter: @escaping @Sendable ([Transcript.Entry]) -> Void,
        historyReplacer: @escaping @Sendable ([Transcript.Entry], UInt64) -> UInt64?
    ) {
        self.historyGetter = historyGetter
        self.historySetter = historySetter
        self.historyReplacer = historyReplacer
    }

    /// The canonical session history after dynamic instructions.
    public var history: Transcript.HistoryView {
        get {
            _accessSessionProperty(keyPath: \.history)
            if SessionPropertyBinding.values === self, let history = SessionPropertyBinding.history {
                return Transcript.HistoryView(history.entries)
            }
            return Transcript.HistoryView(historyGetter())
        }
        set {
            _withSessionPropertyMutation(keyPath: \.history) {
                let newValue = Array(newValue).filter { entry in
                    if case .instructions = entry { return false }
                    return true
                }
                if SessionPropertyBinding.values === self, let history = SessionPropertyBinding.history {
                    precondition(
                        history.isWritable,
                        "Session history is read-only while dynamic instructions or a Tool is active"
                    )
                    history.entries = newValue
                    history.persist(history.entries)
                } else {
                    historySetter(newValue)
                }
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

    /// Returns a history binding for a profile, dynamic instructions, or a Tool.
    ///
    /// The binding's entries reflect the version that `version` holds now.
    /// See ``historyBinding(_:isWritable:protecting:persistingAt:readAt:onChange:)``.
    func historyBinding(
        _ entries: [Transcript.Entry],
        isWritable: Bool,
        protecting protectedEntryKeys: Set<TranscriptEntryKey> = [],
        persistingAt version: Locked<UInt64?>? = nil,
        onChange: (@Sendable ([Transcript.Entry]) -> Void)? = nil
    ) -> SessionHistoryBinding {
        historyBinding(
            entries,
            isWritable: isWritable,
            protecting: protectedEntryKeys,
            persistingAt: version,
            readAt: version?.withLock { $0 },
            onChange: onChange
        )
    }

    /// Returns a history binding whose entries reflect the session history at `snapshotVersion`.
    ///
    /// A writable binding with a `version` persists each write right away,
    /// but only if the session history is still at the version that the binding's entries reflect,
    /// and `version`, which the request's bindings share, is still at that version too.
    /// So a binding whose snapshot is older than another binding's committed write
    /// can't undo that write.
    /// A committed write moves `version` and the binding to the new history version,
    /// and calls `onChange` while it holds `version`'s lock.
    /// Otherwise, the write isn't kept, `version` becomes `nil`
    /// so that later writes aren't kept either,
    /// and ``SessionHistoryBinding/hasConflict`` becomes `true`.
    /// A writable binding without a version keeps its writes to itself.
    func historyBinding(
        _ entries: [Transcript.Entry],
        isWritable: Bool,
        protecting protectedEntryKeys: Set<TranscriptEntryKey> = [],
        persistingAt version: Locked<UInt64?>?,
        readAt snapshotVersion: UInt64?,
        onChange: (@Sendable ([Transcript.Entry]) -> Void)? = nil
    ) -> SessionHistoryBinding {
        let conflict = Locked(false)
        let persist: (@Sendable ([Transcript.Entry]) -> Void)?
        if isWritable, let version {
            let snapshotVersion = Locked(snapshotVersion)
            persist = { [historyReplacer] updated in
                let durable = removingProtectedTranscriptEntries(
                    from: updated,
                    protecting: protectedEntryKeys,
                    in: entries
                )
                let isCommitted = version.withLock { base in
                    guard let expected = snapshotVersion.withLock({ $0 }),
                        base == expected,
                        let committed = historyReplacer(durable, expected)
                    else {
                        base = nil
                        return false
                    }
                    base = committed
                    snapshotVersion.withLock { $0 = committed }
                    onChange?(updated)
                    return true
                }
                if !isCommitted {
                    conflict.withLock { $0 = true }
                }
            }
        } else {
            persist = nil
        }
        return SessionHistoryBinding(
            entries,
            isWritable: isWritable,
            protectedEntryKeys: protectedEntryKeys,
            conflict: conflict,
            persist: persist
        )
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
    @TaskLocal static var history: SessionHistoryBinding?
}

final class SessionHistoryBinding: @unchecked Sendable {
    private let storage: Locked<[Transcript.Entry]>
    let isWritable: Bool
    private let protectedEntryKeys: Set<TranscriptEntryKey>
    private let protectedEntries: [Transcript.Entry]
    private let originalEntries: [Transcript.Entry]
    private let conflict: Locked<Bool>
    private let persistValue: (@Sendable ([Transcript.Entry]) -> Void)?

    init(
        _ entries: [Transcript.Entry],
        isWritable: Bool,
        protectedEntryKeys: Set<TranscriptEntryKey> = [],
        conflict: Locked<Bool> = Locked(false),
        persist: (@Sendable ([Transcript.Entry]) -> Void)? = nil
    ) {
        storage = Locked(
            entries.filter { entry in
                if case .instructions = entry { return false }
                return true
            }
        )
        self.isWritable = isWritable
        self.protectedEntryKeys = protectedEntryKeys
        self.originalEntries = entries
        self.protectedEntries = keyedTranscriptEntries(entries).filter { protectedEntryKeys.contains($0.key) }.map(
            \.entry
        )
        self.conflict = conflict
        self.persistValue = persist
    }

    /// Whether the session history changed after the snapshot that this binding reflects,
    /// so that a write wasn't kept.
    var hasConflict: Bool { conflict.withLock { $0 } }

    var entries: [Transcript.Entry] {
        get { storage.withLock { $0 } }
        set {
            storage.withLock { current in
                current =
                    removingProtectedTranscriptEntries(
                        from: newValue,
                        protecting: protectedEntryKeys,
                        in: originalEntries
                    )
                    + protectedEntries
            }
        }
    }

    func persist(_ entries: [Transcript.Entry]) {
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
                        "SessionProperty can only be accessed while a session profile, dynamic instructions, or a Tool is active"
                    )
                }
                return values[keyPath: keyPath]
            }
            nonmutating set {
                guard let values = SessionPropertyBinding.values else {
                    preconditionFailure(
                        "SessionProperty can only be accessed while a session profile, dynamic instructions, or a Tool is active"
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

    func bindingSessionProperties(_ properties: SessionPropertyValues, history: [Transcript.Entry]) -> any Tool {
        SessionPropertyTool(wrapped: self, properties: properties, history: history)
    }
}

/// A tool that runs another tool with a session's properties bound.
protocol SessionPropertyBoundTool {
    /// The tool as the caller gave it.
    var unboundTool: any Tool { get }
}

private struct SessionPropertyTool<Wrapped: Tool>: Tool, SessionPropertyBoundTool {
    let wrapped: Wrapped
    let properties: SessionPropertyValues
    let history: [Transcript.Entry]

    var name: String { wrapped.name }
    var description: String { wrapped.description }
    var parameters: GenerationSchema { wrapped.parameters }
    var includesSchemaInInstructions: Bool { wrapped.includesSchemaInInstructions }
    var unboundTool: any Tool { wrapped }

    func call(arguments: Wrapped.Arguments) async throws -> Wrapped.Output {
        let entries =
            SessionPropertyBinding.values === properties
            ? SessionPropertyBinding.history?.entries ?? history : history
        let history = SessionHistoryBinding(entries, isWritable: false)
        return try await SessionPropertyBinding.$values.withValue(properties) {
            try await SessionPropertyBinding.$history.withValue(history) {
                try await wrapped.call(arguments: arguments)
            }
        }
    }
}
