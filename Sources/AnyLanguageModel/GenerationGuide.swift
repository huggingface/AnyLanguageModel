import struct Foundation.Decimal
import class Foundation.NSDecimalNumber

/// Guides that control how values are generated.
public struct GenerationGuide<Value>: Sendable {
    var minimumCount: Int?
    var maximumCount: Int?
    var minimum: Double?
    var maximum: Double?
    var stringChoices: [String]?
    var isConstant = false

    /// Creates a guide with no constraints.
    ///
    /// - Note: This initializer is exclusive to AnyLanguageModel.
    ///   It's public for language models outside this module;
    ///   Foundation Models doesn't make it public.
    public init() {}

    init(minimumCount: Int?, maximumCount: Int?) {
        self.minimumCount = minimumCount
        self.maximumCount = maximumCount
    }

    init(minimum: Double?, maximum: Double?) {
        self.minimum = minimum
        self.maximum = maximum
    }

    init(stringChoices: [String], isConstant: Bool) {
        self.stringChoices = stringChoices
        self.isConstant = isConstant
    }

    /// The string choices that `guides` allow, if any of them is a `.constant(_:)` or `.anyOf(_:)` guide.
    ///
    /// As in Foundation Models, a constant takes precedence over `anyOf`,
    /// and a later guide replaces an earlier one of the same kind.
    static func stringChoices(of guides: [GenerationGuide<Value>]) -> (choices: [String], isConstant: Bool)? {
        var constant: String?
        var choices: [String]?
        for guide in guides {
            guard let guideChoices = guide.stringChoices else { continue }
            if guide.isConstant {
                constant = guideChoices.first
            } else {
                choices = guideChoices
            }
        }
        if let constant {
            return ([constant], true)
        }
        return choices.map { ($0, false) }
    }
}

// MARK: - String Guides

extension GenerationGuide where Value == String {

    /// Enforces that the string be precisely the given value.
    public static func constant(_ value: String) -> GenerationGuide<String> {
        GenerationGuide<String>(stringChoices: [value], isConstant: true)
    }

    /// Enforces that the string be one of the provided values.
    public static func anyOf(_ values: [String]) -> GenerationGuide<String> {
        GenerationGuide<String>(stringChoices: values, isConstant: false)
    }

    /// Enforces that the string follows the pattern.
    public static func pattern<Output>(_ regex: Regex<Output>) -> GenerationGuide<String> {
        GenerationGuide<String>()
    }
}

// MARK: - Int Guides

extension GenerationGuide where Value == Int {

    /// Enforces a minimum value.
    ///
    /// Use a `minimum` generation guide --- whose bounds are inclusive --- to ensure the model produces
    /// a value greater than or equal to some minimum value. For example, you can specify that all characters
    /// in your game start at level 1:
    ///
    /// ```swift
    /// @Generable
    /// struct struct GameCharacter {
    ///     @Guide(description: "A creative name appropriate for a fantasy RPG character")
    ///     var name: String
    ///
    ///     @Guide(description: "A level for the character", .minimum(1))
    ///     var level: Int
    /// }
    /// ```
    public static func minimum(_ value: Int) -> GenerationGuide<Int> {
        GenerationGuide<Int>(minimum: Double(value), maximum: nil)
    }

    /// Enforces a maximum value.
    ///
    /// Use a `maximum` generation guide --- whose bounds are inclusive --- to ensure the model produces
    /// a value less than or equal to some maximum value. For example, you can specify that the highest level
    /// a character in your game can achieve is 100:
    ///
    /// ```swift
    /// @Generable
    /// struct struct GameCharacter {
    ///     @Guide(description: "A creative name appropriate for a fantasy RPG character")
    ///     var name: String
    ///
    ///     @Guide(description: "A level for the character", .maximum(100))
    ///     var level: Int
    /// }
    /// ```
    public static func maximum(_ value: Int) -> GenerationGuide<Int> {
        GenerationGuide<Int>(minimum: nil, maximum: Double(value))
    }

    /// Enforces values fall within a range.
    ///
    /// Use a `range` generation guide --- whose bounds are inclusive --- to ensure the model produces a
    /// value that falls within a range. For example, you can specify that the level of characters in your game
    /// are between 1 and 100:
    ///
    /// ```swift
    /// @Generable
    /// struct struct GameCharacter {
    ///     @Guide(description: "A creative name appropriate for a fantasy RPG character")
    ///     var name: String
    ///
    ///     @Guide(description: "A level for the character", .range(1...100))
    ///     var level: Int
    /// }
    /// ```
    public static func range(_ range: ClosedRange<Int>) -> GenerationGuide<Int> {
        GenerationGuide<Int>(minimum: Double(range.lowerBound), maximum: Double(range.upperBound))
    }
}

// MARK: - Float Guides

extension GenerationGuide where Value == Float {

    /// Enforces a minimum value.
    ///
    /// The bounds are inclusive.
    public static func minimum(_ value: Float) -> GenerationGuide<Float> {
        GenerationGuide<Float>(minimum: Double(value), maximum: nil)
    }

    /// Enforces a maximum value.
    ///
    /// The bounds are inclusive.
    public static func maximum(_ value: Float) -> GenerationGuide<Float> {
        GenerationGuide<Float>(minimum: nil, maximum: Double(value))
    }

    /// Enforces values fall within a range.
    public static func range(_ range: ClosedRange<Float>) -> GenerationGuide<Float> {
        GenerationGuide<Float>(minimum: Double(range.lowerBound), maximum: Double(range.upperBound))
    }
}

// MARK: - Decimal Guides

extension GenerationGuide where Value == Decimal {

    /// Enforces a minimum value.
    ///
    /// The bounds are inclusive.
    public static func minimum(_ value: Decimal) -> GenerationGuide<Decimal> {
        GenerationGuide<Decimal>(minimum: NSDecimalNumber(decimal: value).doubleValue, maximum: nil)
    }

    /// Enforces a maximum value.
    ///
    /// The bounds are inclusive.
    public static func maximum(_ value: Decimal) -> GenerationGuide<Decimal> {
        GenerationGuide<Decimal>(minimum: nil, maximum: NSDecimalNumber(decimal: value).doubleValue)
    }

    /// Enforces values fall within a range.
    public static func range(_ range: ClosedRange<Decimal>) -> GenerationGuide<Decimal> {
        GenerationGuide<Decimal>(
            minimum: NSDecimalNumber(decimal: range.lowerBound).doubleValue,
            maximum: NSDecimalNumber(decimal: range.upperBound).doubleValue
        )
    }
}

// MARK: - Double Guides

extension GenerationGuide where Value == Double {

    /// Enforces a minimum value.
    /// The bounds are inclusive.
    public static func minimum(_ value: Double) -> GenerationGuide<Double> {
        GenerationGuide<Double>(minimum: value, maximum: nil)
    }

    /// Enforces a maximum value.
    /// The bounds are inclusive.
    public static func maximum(_ value: Double) -> GenerationGuide<Double> {
        GenerationGuide<Double>(minimum: nil, maximum: value)
    }

    /// Enforces values fall within a range.
    public static func range(_ range: ClosedRange<Double>) -> GenerationGuide<Double> {
        GenerationGuide<Double>(minimum: range.lowerBound, maximum: range.upperBound)
    }
}

// MARK: - Array Guides

extension GenerationGuide {

    /// Enforces a minimum number of elements in the array.
    ///
    /// The bounds are inclusive.
    public static func minimumCount<Element>(_ count: Int) -> GenerationGuide<[Element]>
    where Value == [Element] {
        GenerationGuide<[Element]>(minimumCount: count, maximumCount: nil)
    }

    /// Enforces a maximum number of elements in the array.
    ///
    /// The bounds are inclusive.
    public static func maximumCount<Element>(_ count: Int) -> GenerationGuide<[Element]>
    where Value == [Element] {
        GenerationGuide<[Element]>(minimumCount: nil, maximumCount: count)
    }

    /// Enforces that the number of elements in the array fall within a closed range.
    public static func count<Element>(_ range: ClosedRange<Int>) -> GenerationGuide<[Element]>
    where Value == [Element] {
        GenerationGuide<[Element]>(minimumCount: range.lowerBound, maximumCount: range.upperBound)
    }

    /// Enforces that the array has exactly a certain number elements.
    public static func count<Element>(_ count: Int) -> GenerationGuide<[Element]>
    where Value == [Element] {
        GenerationGuide<[Element]>(minimumCount: count, maximumCount: count)
    }

    /// Enforces a guide on the elements within the array.
    public static func element<Element>(_ guide: GenerationGuide<Element>) -> GenerationGuide<[Element]>
    where Value == [Element] {
        GenerationGuide<[Element]>()
    }
}

// MARK: - Never Array Guides

extension GenerationGuide where Value == [Never] {

    /// Enforces a minimum number of elements in the array.
    ///
    /// Bounds are inclusive.
    ///
    /// - Warning: This overload is only used for macro expansion. Don't call `GenerationGuide<[Never]>.minimumCount(_:)` on your own.
    public static func minimumCount(_ count: Int) -> GenerationGuide<Value> {
        GenerationGuide<Value>(minimumCount: count, maximumCount: nil)
    }

    /// Enforces a maximum number of elements in the array.
    ///
    /// Bounds are inclusive.
    ///
    /// - Warning: This overload is only used for macro expansion. Don't call `GenerationGuide<[Never]>.maximumCount(_:)` on your own.
    public static func maximumCount(_ count: Int) -> GenerationGuide<Value> {
        GenerationGuide<Value>(minimumCount: nil, maximumCount: count)
    }

    /// Enforces that the number of elements in the array fall within a closed range.
    ///
    /// - Warning: This overload is only used for macro expansion. Don't call `GenerationGuide<[Never]>.count(_:)` on your own.
    public static func count(_ range: ClosedRange<Int>) -> GenerationGuide<Value> {
        GenerationGuide<Value>(minimumCount: range.lowerBound, maximumCount: range.upperBound)
    }

    /// Enforces that the array has exactly a certain number elements.
    ///
    /// - Warning: This overload is only used for macro expansion. Don't call `GenerationGuide<[Never]>.count(_:)` on your own.
    public static func count(_ count: Int) -> GenerationGuide<Value> {
        GenerationGuide<Value>(minimumCount: count, maximumCount: count)
    }
}
