import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct AnyLanguageModelMacrosPlugin: CompilerPlugin {
    let providingMacros: [any Macro.Type] = [
        GenerableMacro.self,
        GuideMacro.self,
        SessionPropertyEntryMacro.self,
    ]
}
