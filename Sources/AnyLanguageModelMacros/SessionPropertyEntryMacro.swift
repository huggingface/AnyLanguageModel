import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

/// Synthesizes a `SessionPropertyKey` and forwards the declared property through
/// `SessionPropertyValues`' type-keyed storage.
public struct SessionPropertyEntryMacro: AccessorMacro, PeerMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingAccessorsOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [AccessorDeclSyntax] {
        guard let name = propertyName(in: declaration) else { return [] }
        return [
            "get { self[__Key_\(raw: name).self] }",
            "set { self[__Key_\(raw: name).self] = newValue }",
        ]
    }

    public static func expansion(
        of node: AttributeSyntax,
        providingPeersOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        guard
            let variable = declaration.as(VariableDeclSyntax.self),
            variable.bindings.count == 1,
            let binding = variable.bindings.first,
            let name = binding.pattern.as(IdentifierPatternSyntax.self)?.identifier.text,
            let initializer = binding.initializer?.value
        else { return [] }

        return [
            """
            private struct __Key_\(raw: name): SessionPropertyKey {
                static let defaultValue = \(initializer)
            }
            """
        ]
    }

    private static func propertyName(in declaration: some DeclSyntaxProtocol) -> String? {
        guard
            let variable = declaration.as(VariableDeclSyntax.self),
            variable.bindings.count == 1,
            let binding = variable.bindings.first,
            binding.accessorBlock == nil,
            let name = binding.pattern.as(IdentifierPatternSyntax.self)?.identifier.text,
            binding.initializer != nil
        else { return nil }
        return name
    }
}
