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
            """
            get {
                self._accessSessionProperty(keyPath: \\.\(raw: name))
                return self[__Key_\(raw: name).self]
            }
            """,
            """
            set {
                self._withSessionPropertyMutation(keyPath: \\.\(raw: name)) {
                    self[__Key_\(raw: name).self] = newValue
                }
            }
            """,
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

        // Like Foundation Models,
        // keep the initializer on a stored-looking property for type inference,
        // and let the default-value accessor macro turn it into a computed property,
        // so that each session gets a value of its own.
        let typeAnnotation = binding.typeAnnotation.map { "\($0.trimmed)" } ?? ""
        let defaultValue: DeclSyntax =
            """
            private struct __Key_\(raw: name): SessionPropertyKey {
                @__SessionPropertyEntryDefaultValue
                static var defaultValue\(raw: typeAnnotation) = \(initializer)
            }
            """
        return [defaultValue]
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

/// Turns a session property key's `defaultValue` into a computed property
/// that evaluates its initializer on each access.
public struct SessionPropertyEntryDefaultValueMacro: AccessorMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingAccessorsOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [AccessorDeclSyntax] {
        guard
            let variable = declaration.as(VariableDeclSyntax.self),
            variable.bindings.count == 1,
            let binding = variable.bindings.first,
            binding.accessorBlock == nil,
            let initializer = binding.initializer?.value
        else { return [] }
        return [
            """
            get {
                \(initializer.trimmed)
            }
            """
        ]
    }
}
