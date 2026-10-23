import Foundation
import SwiftParser
import SwiftSyntax
import SwiftSyntaxMacroExpansion
import SwiftSyntaxMacros
import Testing

@testable import AnyLanguageModelMacros

@Suite("Generable macro guides")
struct GenerableMacroGuideTests {
    /// The guides that the `@Generable` expansion of `source` gives its one property.
    private func expandedGuides(of source: String) throws -> String {
        let file = Parser.parse(source: source)
        let structDecl = try #require(file.statements.first?.item.as(StructDeclSyntax.self))
        let attribute = try #require(structDecl.attributes.first?.as(AttributeSyntax.self))
        let members = try GenerableMacro.expansion(
            of: attribute,
            providingMembersOf: structDecl,
            conformingTo: [],
            in: BasicMacroExpansionContext()
        )
        let expansion = members.map(\.description).joined(separator: "\n")
        let line = try #require(
            expansion.split(separator: "\n").first { $0.contains("guides:") }
        )
        return line.trimmingCharacters(in: .whitespaces)
    }

    /// A later bound replaces an earlier one that it contradicts,
    /// as in a schema built from the guides directly.
    @Test(arguments: [
        ("Int", ".minimum(10), .maximum(5)", "guides: [.maximum(5)]"),
        ("Int", ".maximum(5), .minimum(10)", "guides: [.minimum(10)]"),
        ("Double", ".minimum(1.0), .maximum(0.5)", "guides: [.maximum(0.5)]"),
        ("Double", ".maximum(0.5), .minimum(1.0)", "guides: [.minimum(1.0)]"),
        ("Int", ".range(1...9), .maximum(0)", "guides: [.maximum(0)]"),
        ("Int", ".range(1...9), .minimum(10)", "guides: [.minimum(10)]"),
    ])
    func contradictingBoundsKeepTheLaterOne(type: String, guides: String, expected: String) throws {
        let source = """
            @Generable
            struct Sample {
                @Guide(\(guides))
                var value: \(type)
            }
            """
        #expect(try expandedGuides(of: source) == expected)
    }

    /// Bounds in order still expand to the guides they expanded to before.
    @Test(arguments: [
        ("Int", ".minimum(1), .maximum(9)", "guides: [.range(1...9)]"),
        ("Int", ".maximum(9), .minimum(1)", "guides: [.range(1...9)]"),
        ("Int", ".minimum(5), .maximum(5)", "guides: [.range(5...5)]"),
        ("Int", ".range(1...9)", "guides: [.range(1...9)]"),
        ("Int", ".minimum(-3)", "guides: [.minimum(-3)]"),
        ("Int", ".maximum(3)", "guides: [.maximum(3)]"),
        ("Double", ".minimum(0.5), .maximum(1.5)", "guides: [.range(0.5...1.5)]"),
        ("Double", ".range(0.0...1.0)", "guides: [.range(0.0...1.0)]"),
    ])
    func boundsInOrderAreUnchanged(type: String, guides: String, expected: String) throws {
        let source = """
            @Generable
            struct Sample {
                @Guide(\(guides))
                var value: \(type)
            }
            """
        #expect(try expandedGuides(of: source) == expected)
    }
}
