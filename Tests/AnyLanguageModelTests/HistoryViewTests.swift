import Testing

import AnyLanguageModel

@Suite("HistoryView collection contract")
struct HistoryViewTests {
    private func entry(_ text: String) -> Transcript.Entry {
        .prompt(.init(segments: [.text(.init(content: text))]))
    }

    @Test func mutableSlicesPreserveIndicesAndValueSemantics() {
        let first = entry("first")
        let second = entry("second")
        let third = entry("third")
        var history: Transcript.HistoryView = [first, second, third]
        let original = history
        let middle = history.index(after: history.startIndex)
        var slice: Transcript.HistoryView = history[middle ..< history.endIndex]

        #expect(slice.startIndex == middle)
        #expect(slice.endIndex == history.endIndex)
        #expect(slice.distance(from: slice.startIndex, to: slice.endIndex) == 2)
        #expect(slice.index(before: slice.endIndex) == history.index(history.startIndex, offsetBy: 2))
        #expect(Array(slice.indices) == [middle, middle.advanced(by: 1)])
        #expect(slice.startIndex.distance(to: slice.endIndex) == 2)
        slice[slice.startIndex] = first
        #expect(Array(history) == [first, second, third])
        history[middle ..< history.endIndex] = slice
        #expect(Array(history) == [first, first, third])
        #expect(Array(original) == [first, second, third])
    }

    @Test func rangeReplacementAndCollectionInitializers() {
        let first = entry("first")
        let second = entry("second")
        let third = entry("third")
        var history = Transcript.HistoryView([first, second])
        history.replaceSubrange(history.startIndex ..< history.index(after: history.startIndex), with: [third, first])
        #expect(Array(history) == [third, first, second])
        history.append(third)
        history.append(contentsOf: [second])
        history.removeFirst()
        #expect(Array(history) == [first, second, third, second])
        history.removeAll()
        #expect(history.isEmpty)
    }
}
