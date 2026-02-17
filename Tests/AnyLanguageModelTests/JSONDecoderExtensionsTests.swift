import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("JSONDecoder Extensions")
struct JSONDecoderExtensionsTests {
    private struct Payload: Decodable {
        let date: Date
    }

    @Test func iso8601WithFractionalSecondsDecodesFractionalDates() throws {
        let json = #"{"date":"2026-02-17T12:34:56.789Z"}"#.data(using: .utf8)!
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601WithFractionalSeconds

        let payload = try decoder.decode(Payload.self, from: json)

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let expected = formatter.date(from: "2026-02-17T12:34:56.789Z")!
        #expect(payload.date == expected)
    }

    @Test func iso8601WithFractionalSecondsFallsBackToNonFractionalDates() throws {
        let json = #"{"date":"2026-02-17T12:34:56Z"}"#.data(using: .utf8)!
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601WithFractionalSeconds

        let payload = try decoder.decode(Payload.self, from: json)

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        let expected = formatter.date(from: "2026-02-17T12:34:56Z")!
        #expect(payload.date == expected)
    }

    @Test func iso8601WithFractionalSecondsThrowsDataCorruptedForInvalidDates() {
        let json = #"{"date":"not-a-date"}"#.data(using: .utf8)!
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601WithFractionalSeconds

        do {
            _ = try decoder.decode(Payload.self, from: json)
            Issue.record("Expected decode to fail for invalid date")
        } catch let error as DecodingError {
            if case .dataCorrupted = error {
                #expect(Bool(true))
            } else {
                Issue.record("Expected dataCorrupted, got \(error)")
            }
        } catch {
            Issue.record("Expected DecodingError, got \(error)")
        }
    }
}
