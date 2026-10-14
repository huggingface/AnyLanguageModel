// Foundation Models' own `@Generable` enums,
// declared like the enums in `GenerableEnumPayloadTests`.
// This file doesn't import AnyLanguageModel,
// whose macro expansions would make names like `GeneratedContent` ambiguous.

#if canImport(FoundationModels) && compiler(>=6.4) && !os(tvOS)
    import FoundationModels

    @available(macOS 26.0, iOS 26.0, watchOS 27.0, visionOS 26.0, *)
    enum FoundationModelsEnums {
        @Generable
        enum Operand: Equatable {
            case name(String)
            case number(Double)
        }

        @Generable(description: "One thing a clause does.")
        enum Step: Equatable {
            case set(field: String, to: Operand)
            case add(Operand, to: String)
            case pair(Int, Int)
            case note(String?)
            case reset
        }

        typealias MaybeText = String?

        @Generable
        enum Annotation: Equatable {
            case note(String?)
            case alias(MaybeText)
            case generic(Optional<Int>)
            case qualified(Swift.Optional<String>)
        }
    }
#endif
