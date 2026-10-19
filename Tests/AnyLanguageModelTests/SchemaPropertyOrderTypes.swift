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
