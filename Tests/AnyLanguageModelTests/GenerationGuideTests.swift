import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("GenerationGuide")
struct GenerationGuideTests {
    @Test func stringFactoriesAreCallable() {
        _ = GenerationGuide<String>.constant("fixed")
        _ = GenerationGuide<String>.anyOf(["red", "green", "blue"])
        _ = GenerationGuide<String>.pattern(#/^[a-z]+$/#)
    }

    @Test func intFactoriesSetBounds() {
        let minimum = GenerationGuide<Int>.minimum(1)
        let maximum = GenerationGuide<Int>.maximum(10)
        let range = GenerationGuide<Int>.range(2 ... 8)

        #expect(minimum.minimum == 1)
        #expect(minimum.maximum == nil)
        #expect(maximum.minimum == nil)
        #expect(maximum.maximum == 10)
        #expect(range.minimum == 2)
        #expect(range.maximum == 8)
    }

    @Test func floatFactoriesSetBounds() {
        let minimum = GenerationGuide<Float>.minimum(1.25)
        let maximum = GenerationGuide<Float>.maximum(9.75)
        let range = GenerationGuide<Float>.range(2.5 ... 8.5)

        #expect(minimum.minimum == 1.25)
        #expect(maximum.maximum == 9.75)
        #expect(range.minimum == 2.5)
        #expect(range.maximum == 8.5)
    }

    @Test func doubleFactoriesSetBounds() {
        let minimum = GenerationGuide<Double>.minimum(0.1)
        let maximum = GenerationGuide<Double>.maximum(0.9)
        let range = GenerationGuide<Double>.range(0.2 ... 0.8)

        #expect(minimum.minimum == 0.1)
        #expect(maximum.maximum == 0.9)
        #expect(range.minimum == 0.2)
        #expect(range.maximum == 0.8)
    }

    @Test func decimalFactoriesSetBounds() {
        let minimum = GenerationGuide<Decimal>.minimum(Decimal(string: "1.5")!)
        let maximum = GenerationGuide<Decimal>.maximum(Decimal(string: "9.5")!)
        let range = GenerationGuide<Decimal>.range(Decimal(string: "2.5")! ... Decimal(string: "8.5")!)

        #expect(minimum.minimum == 1.5)
        #expect(maximum.maximum == 9.5)
        #expect(range.minimum == 2.5)
        #expect(range.maximum == 8.5)
    }

    @Test func arrayFactoriesSetCountBounds() {
        let minimum = GenerationGuide<[String]>.minimumCount(1)
        let maximum = GenerationGuide<[String]>.maximumCount(5)
        let range = GenerationGuide<[String]>.count(2 ... 4)
        let exact = GenerationGuide<[String]>.count(3)
        let element = GenerationGuide<[String]>.element(.constant("x"))

        #expect(minimum.minimumCount == 1)
        #expect(maximum.maximumCount == 5)
        #expect(range.minimumCount == 2)
        #expect(range.maximumCount == 4)
        #expect(exact.minimumCount == 3)
        #expect(exact.maximumCount == 3)
        #expect(element.minimumCount == nil)
        #expect(element.maximumCount == nil)
    }
}
