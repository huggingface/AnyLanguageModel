/// The values a number may take, named after Swift's ranges: `...`, `v...`, `...v` and `a...b`.
///
/// Made only from bounds in order, so a minimum above its maximum can't be represented.
enum Bounds<Value: Comparable & Sendable>: Sendable, Equatable {
    case unbounded
    case from(Value)
    case through(Value)
    case closed(ClosedRange<Value>)

    /// The bounds from `minimum` to `maximum`, either of them absent,
    /// or `nil` when `minimum` is above `maximum`, which no value is within.
    init?(minimum: Value?, maximum: Value?) {
        switch (minimum, maximum) {
        case (nil, nil): self = .unbounded
        case (let minimum?, nil): self = .from(minimum)
        case (nil, let maximum?): self = .through(maximum)
        case (let minimum?, let maximum?):
            guard minimum <= maximum else { return nil }
            self = .closed(minimum ... maximum)
        }
    }

    /// These bounds with `minimum` as their minimum,
    /// dropping a maximum below it.
    func replacing(minimum: Value) -> Bounds {
        Bounds(minimum: minimum, maximum: maximum.flatMap { $0 >= minimum ? $0 : nil })!
    }

    /// These bounds with `maximum` as their maximum,
    /// dropping a minimum above it.
    func replacing(maximum: Value) -> Bounds {
        Bounds(minimum: minimum.flatMap { $0 <= maximum ? $0 : nil }, maximum: maximum)!
    }

    var minimum: Value? {
        switch self {
        case .from(let minimum): minimum
        case .closed(let range): range.lowerBound
        case .unbounded, .through: nil
        }
    }

    var maximum: Value? {
        switch self {
        case .through(let maximum): maximum
        case .closed(let range): range.upperBound
        case .unbounded, .from: nil
        }
    }

    func contains(_ value: Value) -> Bool {
        clamped(value) == value
    }

    /// `value`, or the bound it's past.
    func clamped(_ value: Value) -> Value {
        var clamped = value
        if let minimum { clamped = max(clamped, minimum) }
        if let maximum { clamped = min(clamped, maximum) }
        return clamped
    }
}
