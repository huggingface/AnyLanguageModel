import Foundation

@testable import AnyLanguageModel

final class MockTokenCapture {
    private(set) var tokens: [Int] = []
    var tokenTextCalls = 0
    var specialTokenCalls = 0
    var tokenToText: [Int: String]

    init(tokenToText: [Int: String]) {
        self.tokenToText = tokenToText
    }

    func record(token: Int) {
        tokens.append(token)
    }

    var decodedText: String {
        tokens.compactMap { tokenToText[$0] }.joined()
    }
}

struct MockTokenBackend: TokenBackend {
    var tokenToText: [Int: String]
    var textToTokens: [String: [Int]]
    var specialTokens: Set<Int>
    var endTokens: Set<Int>
    var eosToken: Int
    var vocabSize: Int
    var remainingTokens: Int
    let totalTokenBudget: Int
    var capture: MockTokenCapture
    var samplingQueue: [Int]

    init(
        tokenToText: [Int: String],
        textToTokens: [String: [Int]] = [:],
        specialTokens: Set<Int> = [],
        eosToken: Int,
        endTokens: Set<Int>,
        maximumTokens: Int,
        samplingQueue: [Int] = []
    ) {
        self.tokenToText = tokenToText
        self.textToTokens = textToTokens
        self.specialTokens = specialTokens
        self.eosToken = eosToken
        self.endTokens = endTokens
        self.totalTokenBudget = maximumTokens
        self.remainingTokens = maximumTokens
        self.vocabSize = max((tokenToText.keys.max() ?? 0) + 1, 1)
        self.capture = MockTokenCapture(tokenToText: tokenToText)
        self.samplingQueue = samplingQueue
    }

    func tokenize(_ text: String) throws -> [Int] {
        if let tokens = textToTokens[text] {
            return tokens
        }
        if let token = tokenToText.first(where: { $0.value == text })?.key {
            return [token]
        }
        return []
    }

    func tokenText(_ token: Int) -> String? {
        capture.tokenTextCalls += 1
        return tokenToText[token]
    }

    func isSpecialToken(_ token: Int) -> Bool {
        capture.specialTokenCalls += 1
        return specialTokens.contains(token)
    }

    mutating func decode(_ token: Int) async throws {
        capture.record(token: token)
        remainingTokens -= 1
    }

    mutating func sample(from allowedTokens: Set<Int>) async throws -> Int {
        guard !allowedTokens.isEmpty else {
            throw ConstrainedGenerationError.tokenizationFailed
        }
        if !samplingQueue.isEmpty {
            let token = samplingQueue.removeFirst()
            guard allowedTokens.contains(token) else {
                throw ConstrainedGenerationError.tokenizationFailed
            }
            return token
        }
        guard let token = allowedTokens.min() else {
            throw ConstrainedGenerationError.tokenizationFailed
        }
        return token
    }
}
