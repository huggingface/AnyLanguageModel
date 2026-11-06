import SwiftParser
import SwiftSyntax
import SwiftSyntaxMacroExpansion
import SwiftSyntaxMacros
import Testing

@testable import AnyLanguageModelMacros

@Suite("Session property entry macro")
struct SessionPropertyEntryMacroTests {
    /// The variable declaration in `source` and its first attribute.
    private func variable(in source: String) throws -> (VariableDeclSyntax, AttributeSyntax) {
        let file = Parser.parse(source: source)
        let variable = try #require(file.statements.first?.item.as(VariableDeclSyntax.self))
        let attribute = try #require(variable.attributes.first?.as(AttributeSyntax.self))
        return (variable, attribute)
    }

    /// As in Foundation Models, the key's default value is computed,
    /// so that each session gets its own value.
    @Test(arguments: [
        ("@SessionPropertyEntry var counter = Counter()", "static var defaultValue = Counter()"),
        ("@SessionPropertyEntry var counter: Counter = .init()", "static var defaultValue: Counter = .init()"),
    ])
    func keyDefaultValueIsComputed(source: String, declaration: String) throws {
        let (variable, attribute) = try variable(in: source)
        let peers = try SessionPropertyEntryMacro.expansion(
            of: attribute,
            providingPeersOf: variable,
            in: BasicMacroExpansionContext()
        )
        let key = try #require(peers.first?.as(StructDeclSyntax.self))
        let defaultValue = try #require(key.memberBlock.members.first?.decl.as(VariableDeclSyntax.self))

        #expect(key.name.text == "__Key_counter")
        #expect(defaultValue.attributes.trimmedDescription == "@__SessionPropertyEntryDefaultValue")
        #expect(defaultValue.trimmedDescription.hasSuffix(declaration))
    }

    @Test func defaultValueAccessorEvaluatesTheInitializer() throws {
        let (variable, attribute) = try variable(
            in: "@__SessionPropertyEntryDefaultValue static var defaultValue = Counter()"
        )
        let accessors = try SessionPropertyEntryDefaultValueMacro.expansion(
            of: attribute,
            providingAccessorsOf: variable,
            in: BasicMacroExpansionContext()
        )

        #expect(accessors.count == 1)
        #expect(accessors.first?.accessorSpecifier.tokenKind == .keyword(.get))
        #expect(accessors.first?.body?.statements.trimmedDescription == "Counter()")
    }
}
