import Foundation

/// A prompt from a person to the model.
///
/// Prompts can contain content written by you, an outside source, or input directly from people using
/// your app. You can initialize a `Prompt` from a string literal:
///
/// ```swift
/// let prompt = Prompt("What are miniature schnauzers known for?")
/// ```
///
/// Use ``PromptBuilder`` to dynamically control the prompt's content based on your app's state. The
/// code below shows if the Boolean is `true`, the prompt includes a second line of text:
///
/// ```swift
/// let responseShouldRhyme = true
/// let prompt = Prompt {
///     "Answer the following question from the user: \(userInput)"
///     if responseShouldRhyme {
///         "Your response MUST rhyme!"
///     }
/// }
/// ```
///
/// If your prompt includes input from people, consider wrapping the input in a string template with your
/// own prompt to better steer the model's response. For more information on handling inputs in your
/// prompts, see <doc:improving-safety-from-generative-model-output>.
public struct Prompt: Sendable {
    enum Component: Sendable {
        case text(Transcript.TextSegment)
        case image(id: String, content: ImageAttachmentContent)
    }

    let components: [Component]

    /// Creates an instance with the content you specify.
    public init(_ representable: some PromptRepresentable) {
        switch representable {
        case let prompt as Prompt:
            self = prompt
        case let string as String:
            self.init(content: string)
        default:
            self = representable.promptRepresentation
        }
    }

    init(content: String) {
        components = [.text(.init(content: content))]
    }

    init(components: [Component]) {
        self.components = components
    }

    func makeTranscriptSegments() throws -> [Transcript.Segment] {
        try components.map { component in
            switch component {
            case .text(let text):
                return .text(text)
            case .image(let id, let content):
                return .image(try content.makeSegment(id: id))
            }
        }
    }

    static func joining(_ prompts: [Prompt], trim: Bool = false) -> Prompt {
        var components: [Component] = []
        for prompt in prompts {
            for component in prompt.components {
                if case .text(let text) = component, let last = components.last,
                    case .text(let previous) = last
                {
                    components[components.count - 1] = .text(
                        .init(
                            id: previous.id,
                            content: previous.content + "\n" + text.content
                        )
                    )
                } else {
                    components.append(component)
                }
            }
        }

        if trim {
            if let first = components.first, case .text(let text) = first {
                components[0] = .text(
                    .init(
                        id: text.id,
                        content: text.content.replacingOccurrences(
                            of: #"^\s+"#,
                            with: "",
                            options: .regularExpression
                        )
                    )
                )
            }
            if let last = components.last, case .text(let text) = last {
                components[components.count - 1] = .text(
                    .init(
                        id: text.id,
                        content: text.content.replacingOccurrences(
                            of: #"\s+$"#,
                            with: "",
                            options: .regularExpression
                        )
                    )
                )
            }
        }

        return Prompt(components: components)
    }

    public init(@PromptBuilder _ content: () throws -> Prompt) rethrows {
        self = try content()
    }
}

// MARK: - CustomStringConvertible

extension Prompt: CustomStringConvertible {
    public var description: String {
        components.map { component in
            switch component {
            case .text(let text):
                return text.content
            case .image:
                return "<image>"
            }
        }.joined(separator: "\n")
    }
}

// MARK: - PromptBuilder

@resultBuilder
public struct PromptBuilder {
    public static func buildBlock<each P>(_ components: repeat each P) -> Prompt
    where repeat each P: PromptRepresentable {
        var parts: [Prompt] = []
        repeat parts.append((each components).promptRepresentation)
        return Prompt.joining(parts, trim: true)
    }

    public static func buildExpression<P>(_ expression: P) -> P where P: PromptRepresentable {
        return expression
    }

    public static func buildArray(_ prompts: [some PromptRepresentable]) -> Prompt {
        Prompt.joining(prompts.map(\.promptRepresentation))
    }

    public static func buildOptional(_ component: Prompt?) -> Prompt {
        return component ?? Prompt(content: "")
    }

    public static func buildEither(first component: some PromptRepresentable) -> Prompt {
        return component.promptRepresentation
    }

    public static func buildEither(second component: some PromptRepresentable) -> Prompt {
        return component.promptRepresentation
    }

    public static func buildLimitedAvailability(_ prompt: some PromptRepresentable) -> Prompt {
        return prompt.promptRepresentation
    }
}

// MARK: - PromptRepresentable

/// A protocol that represents a prompt.
public protocol PromptRepresentable {
    /// An instance that represents a prompt.
    var promptRepresentation: Prompt { get }
}

// MARK: - Default Implementations

extension Prompt: PromptRepresentable {
    /// An instance that represents a prompt.
    public var promptRepresentation: Prompt { self }
}

// MARK: - Standard Library Extensions

extension String: PromptRepresentable {
    /// An instance that represents a prompt.
    public var promptRepresentation: Prompt {
        Prompt(content: self)
    }
}

extension Array: PromptRepresentable where Element: PromptRepresentable {
    /// An instance that represents a prompt.
    public var promptRepresentation: Prompt {
        Prompt.joining(map(\.promptRepresentation))
    }
}
