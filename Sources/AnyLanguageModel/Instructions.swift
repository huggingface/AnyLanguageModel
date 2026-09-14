/// Instructions define the model's intended behavior on prompts.
///
/// Instructions are typically provided by you to define the role and behavior of the model. In the code below,
/// the instructions specify that the model replies with topics rather than, for example, a recipe:
///
/// ```swift
/// let instructions = """
///     Suggest related topics. Keep them concise (three to seven words) and \
///     make sure they build naturally from the person's topic.
///     """
///
/// let session = LanguageModelSession(instructions: instructions)
///
/// let prompt = "Making homemade bread"
/// let response = try await session.respond(to: prompt)
/// ```
///
/// Apple trains the model to obey instructions over any commands it receives in prompts, so don't include
/// untrusted content in instructions. For more on how instructions impact generation quality and safety,
/// see <doc:improving-safety-from-generative-model-output>.
public struct Instructions: Sendable {
    private let content: String

    /// Creates an instance with the content you specify.
    public init(_ representable: some InstructionsRepresentable) {
        switch representable {
        case let instructions as Instructions:
            self = instructions
        case let string as String:
            self.init(content: string)
        default:
            self.init(content: representable.instructionsRepresentation.content)
        }
    }

    init(content: String) {
        self.content = content
    }

    public init(@InstructionsBuilder _ content: () throws -> Instructions) rethrows {
        self = try content()
    }
}

// MARK: - CustomStringConvertible

extension Instructions: CustomStringConvertible {
    public var description: String { content }
}

// MARK: - InstructionsBuilder

@resultBuilder
public struct InstructionsBuilder {
    public static func buildBlock<each I>(_ components: repeat each I) -> Instructions
    where repeat each I: InstructionsRepresentable {
        var parts: [String] = []
        repeat parts.append((each components).instructionsRepresentation.description)
        let combinedText = parts.joined(separator: "\n")
        return Instructions(content: combinedText.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    public static func buildExpression<I>(_ expression: I) -> I where I: InstructionsRepresentable {
        return expression
    }

    public static func buildArray(_ instructions: [some InstructionsRepresentable]) -> Instructions {
        let combinedText = instructions.map {
            $0.instructionsRepresentation.description
        }.joined(separator: "\n")
        return Instructions(content: combinedText)
    }

    public static func buildOptional(_ instructions: Instructions?) -> Instructions {
        return instructions ?? Instructions(content: "")
    }

    public static func buildEither(first component: some InstructionsRepresentable) -> Instructions {
        return component.instructionsRepresentation
    }

    public static func buildEither(second component: some InstructionsRepresentable) -> Instructions {
        return component.instructionsRepresentation
    }

    public static func buildLimitedAvailability(_ instructions: some InstructionsRepresentable)
        -> Instructions
    {
        return instructions.instructionsRepresentation
    }
}

/// Conforming types represent instructions.
public protocol InstructionsRepresentable {
    /// An instance that represents the instructions.
    var instructionsRepresentation: Instructions { get }
}

// MARK: - Default Implementations

extension Instructions: InstructionsRepresentable {
    /// An instance that represents the instructions.
    public var instructionsRepresentation: Instructions { self }
}

// MARK: - Standard Library Extensions

extension String: InstructionsRepresentable {
    /// An instance that represents the instructions.
    public var instructionsRepresentation: Instructions {
        Instructions(content: self)
    }
}

extension Array: InstructionsRepresentable where Element: InstructionsRepresentable {
    /// An instance that represents the instructions.
    public var instructionsRepresentation: Instructions {
        let combined = self.map { $0.instructionsRepresentation.description }
            .joined(separator: "\n")
        return Instructions(content: combined)
    }
}
