/// A tool that a model can call to gather information at runtime or perform side effects.
///
/// Tool calling gives the model the ability to call your code to incorporate
/// up-to-date information like recent events and data from your app. A tool
/// includes a name and a description that the framework puts in the prompt to let
/// the model decide when and how often to call your tool.
///
/// A `Tool` defines a ``call(arguments:)`` method that takes arguments that conforms to
/// ``ConvertibleFromGeneratedContent``, and returns an output of any type that conforms to
/// ``PromptRepresentable``, allowing the model to understand and reason about in subsequent
/// interactions. Typically, ``Output`` is a `String` or any ``Generable`` types.
///
/// ```swift
/// struct FindContacts: Tool {
///     let name = "findContacts"
///     let description = "Finds a specific number of contacts"
///
///     @Generable
///     struct Arguments {
///         @Guide(description: "The number of contacts to get", .range(1...10))
///         let count: Int
///     }
///
///     func call(arguments: Arguments) async throws -> [String] {
///         var contacts: [CNContact] = []
///         // Fetch a number of contacts using the arguments.
///         let formattedContacts = contacts.map {
///             "\($0.givenName) \($0.familyName)"
///         }
///         return formattedContacts
///     }
/// }
/// ```
///
/// Tools must conform to <doc://com.apple.documentation/documentation/swift/sendable>
/// so the framework can run them concurrently. If the model needs to pass the output
/// of one tool as the input to another, it executes back-to-back tool calls.
///
/// You control the life cycle of your tool, so you can track the state of it between
/// calls to the model. For example, you might store a list of database records that
/// you don't want to reuse between tool calls.
public protocol Tool<Arguments, Output>: Sendable {

    /// The output that this tool produces for the language model to reason about in subsequent
    /// interactions.
    ///
    /// Typically output is either a ``String`` or a ``Generable`` type.
    associatedtype Output: PromptRepresentable

    /// The arguments that this tool should accept.
    ///
    /// Typically arguments are either a ``Generable`` type or ``GeneratedContent``.
    associatedtype Arguments: ConvertibleFromGeneratedContent

    /// A unique name for the tool, such as "get_weather", "toggleDarkMode", or "search contacts".
    var name: String { get }

    /// A natural language description of when and how to use the tool.
    var description: String { get }

    /// A schema for the parameters this tool accepts.
    var parameters: GenerationSchema { get }

    /// If true, the model's name, description, and parameters schema will be injected
    /// into the instructions of sessions that leverage this tool.
    ///
    /// The default implementation is `true`
    ///
    /// - Note: This should only be `false` if the model has been trained to have
    /// innate knowledge of this tool. For zero-shot prompting, it should always be `true`.
    var includesSchemaInInstructions: Bool { get }

    /// A language model will call this method when it wants to leverage this tool.
    ///
    /// If errors are throw in the body of this method, they will be wrapped in a
    /// ``LanguageModelSession.ToolCallError`` and rethrown at the call site
    /// of ``LanguageModelSession.respond(to:)``.
    ///
    /// - Note: This method may be invoked concurrently with itself or with other tools.
    func call(arguments: Self.Arguments) async throws -> Self.Output
}

// MARK: - Default Implementations

extension Tool {
    /// A unique name for the tool, such as "get_weather", "toggleDarkMode", or "search contacts".
    public var name: String {
        String(describing: Self.self)
    }

    /// If true, the model's name, description, and parameters schema will be injected
    /// into the instructions of sessions that leverage this tool.
    ///
    /// The default implementation is `true`
    ///
    /// - Note: This should only be `false` if the model has been trained to have
    /// innate knowledge of this tool. For zero-shot prompting, it should always be `true`.
    public var includesSchemaInInstructions: Bool {
        true
    }
}

extension Tool where Self.Arguments: Generable {
    /// A schema for the parameters this tool accepts.
    public var parameters: GenerationSchema {
        Arguments.generationSchema
    }
}

// MARK: - Helpers

extension Tool {
    /// Compatibility integration SPI for provider adapters outside this module.
    /// This is not part of Apple's public Tool contract.
    @_spi(Compatibility)
    public func makeOutputSegments(from arguments: GeneratedContent) async throws -> [Transcript.Segment] {
        let parsedArguments = try Arguments(arguments)
        let output = try await call(arguments: parsedArguments)

        if let structured = output as? any ConvertibleToGeneratedContent {
            let content = structured.generatedContent
            let segment = Transcript.Segment.structure(.init(source: name, content: content))
            return [segment]
        }

        if let stringOutput = output as? String {
            return [Transcript.Segment.text(.init(content: stringOutput))]
        }

        return try output.promptRepresentation.makeTranscriptSegments()
    }
}
