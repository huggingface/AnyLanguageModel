import Foundation
import Testing

@testable import AnyLanguageModel

/// Properties declared out of alphabetical order, as a model must generate them: what an item
/// is first, then what it holds.
@Generable
struct OrderedLeaf {
    var kind: String
    var words: String
}

@Generable
struct OrderedItem {
    var kind: String
    var words: String
    var items: [OrderedLeaf]
}

/// Whether `names` first appear in `text` in the order given.
func appearInOrder(_ names: [String], in text: String) -> Bool {
    let positions = names.compactMap { text.range(of: "\"\($0)\"")?.lowerBound }
    return positions.count == names.count && positions == positions.sorted()
}
