/// A declarative collection of instructions and tools that a session resolves
/// immediately before each request to a language model.
///
/// Compose values in ``body`` with ``DynamicInstructionsBuilder``. The session
/// evaluates the body again for every model request, including requests that
/// continue a response after tool execution.
@_typeEraser(AnyDynamicInstructions)
public protocol DynamicInstructions {
    associatedtype Body: DynamicInstructions

    @DynamicInstructionsBuilder
    var body: Body { get }
}

/// Builds declarative dynamic instructions from instructions, tools, nested
/// dynamic instructions, and conditional content.
@resultBuilder
public struct DynamicInstructionsBuilder {
    public static func buildExpression<T>(_ expression: T) -> some DynamicInstructions where T: Tool {
        DynamicTool(expression)
    }

    public static func buildExpression<T>(_ expression: T) -> T where T: DynamicInstructions {
        expression
    }

    public static func buildExpression(_ tools: [any Tool]) -> some DynamicInstructions {
        DynamicInstructionsForEach(tools, id: \.name) { tool in
            AnyDynamicInstructions(DynamicTool(tool))
        }
    }

    @_disfavoredOverload
    public static func buildBlock<each Content>(
        _ contents: repeat each Content
    ) -> TupleDynamicInstructions<repeat each Content>
    where repeat each Content: DynamicInstructions {
        TupleDynamicInstructions(repeat each contents)
    }

    public static func buildBlock<T>(_ content: T) -> T where T: DynamicInstructions {
        content
    }

    public static func buildBlock() -> EmptyDynamicInstructions {
        EmptyDynamicInstructions()
    }

    public static func buildEither<TrueContent, FalseContent>(
        first content: TrueContent
    ) -> ConditionalDynamicInstructions<TrueContent, FalseContent>
    where TrueContent: DynamicInstructions, FalseContent: DynamicInstructions {
        ConditionalDynamicInstructions(.trueContent(content))
    }

    public static func buildEither<TrueContent, FalseContent>(
        second content: FalseContent
    ) -> ConditionalDynamicInstructions<TrueContent, FalseContent>
    where TrueContent: DynamicInstructions, FalseContent: DynamicInstructions {
        ConditionalDynamicInstructions(.falseContent(content))
    }

    public static func buildOptional<Content>(_ content: Content?) -> Content?
    where Content: DynamicInstructions {
        content
    }

    public static func buildLimitedAvailability(
        _ content: some DynamicInstructions
    ) -> AnyDynamicInstructions {
        AnyDynamicInstructions(content)
    }
}

/// A type-erased dynamic-instructions value.
public struct AnyDynamicInstructions: DynamicInstructions {
    public typealias Body = Never

    fileprivate let resolveValue: () -> ResolvedDynamicInstructions

    public init(_ dynamicInstructions: any DynamicInstructions) {
        resolveValue = { resolveDynamicInstructions(dynamicInstructions) }
    }

    public init(erasing dynamicInstructions: some DynamicInstructions) {
        self.init(dynamicInstructions)
    }

    public var body: Never {
        fatalError("AnyDynamicInstructions has no body")
    }

    func resolveForRequest() -> ResolvedDynamicInstructions {
        resolveValue()
    }
}

/// A dynamic-instructions value that contains an ordered tuple of components.
public struct TupleDynamicInstructions<each Content>: DynamicInstructions
where repeat each Content: DynamicInstructions {
    public typealias Body = Never

    fileprivate let contents: (repeat each Content)

    public init(_ contents: repeat each Content) {
        self.contents = (repeat each contents)
    }

    public var body: Never {
        fatalError("TupleDynamicInstructions has no body")
    }
}

/// A dynamic-instructions value that contains one of two branches.
public struct ConditionalDynamicInstructions<TrueContent, FalseContent>: DynamicInstructions
where TrueContent: DynamicInstructions, FalseContent: DynamicInstructions {
    public enum Branch {
        case trueContent(TrueContent)
        case falseContent(FalseContent)
    }

    public typealias Body = Never

    fileprivate let branch: Branch

    public init(_ branch: Branch) {
        self.branch = branch
    }

    public var body: Never {
        fatalError("ConditionalDynamicInstructions has no body")
    }
}

extension Optional: DynamicInstructions where Wrapped: DynamicInstructions {
    public typealias Body = Never

    public var body: Never {
        fatalError("Optional dynamic instructions have no body")
    }
}

extension Never: DynamicInstructions {
    public typealias Body = Never

    public var body: Never { self }
}

/// An empty dynamic-instructions value.
public struct EmptyDynamicInstructions: DynamicInstructions, Sendable {
    public typealias Body = Never

    public init() {}

    public var body: Never {
        fatalError("EmptyDynamicInstructions has no body")
    }
}

/// Builds dynamic instructions from a collection.
public struct DynamicInstructionsForEach<Data, ID, Content>: DynamicInstructions
where Data: RandomAccessCollection, ID: Hashable, Content: DynamicInstructions {
    public typealias Body = Never

    fileprivate let data: Data
    fileprivate let id: KeyPath<Data.Element, ID>
    fileprivate let content: (Data.Element) -> Content

    public init(
        _ data: Data,
        id: KeyPath<Data.Element, ID>,
        @DynamicInstructionsBuilder content: @escaping (Data.Element) -> Content
    ) {
        self.data = data
        self.id = id
        self.content = content
    }

    public var body: Never {
        fatalError("DynamicInstructionsForEach has no body")
    }
}

extension DynamicInstructionsForEach where ID == Data.Element.ID, Data.Element: Identifiable {
    public init(
        _ data: Data,
        @DynamicInstructionsBuilder content: @escaping (Data.Element) -> Content
    ) {
        self.init(data, id: \.id, content: content)
    }
}

extension DynamicInstructions {
    public typealias ForEach = DynamicInstructionsForEach
}

extension Instructions: DynamicInstructions {
    public var body: some DynamicInstructions {
        EmptyDynamicInstructions()
    }
}

struct ResolvedDynamicInstructions: Sendable {
    let instructions: Instructions?
    let tools: [any Tool]

    fileprivate init(instructions: Instructions?, tools: [any Tool]) {
        self.instructions = instructions
        self.tools = tools
    }

    fileprivate static let empty = Self(instructions: nil, tools: [])

    fileprivate func appending(_ other: Self) -> Self {
        let combinedInstructions: Instructions?
        switch (instructions, other.instructions) {
        case (nil, nil):
            combinedInstructions = nil
        case (let instructions?, nil), (nil, let instructions?):
            combinedInstructions = instructions
        case (let first?, let second?):
            combinedInstructions = Instructions {
                first
                second
            }
        }
        return Self(
            instructions: combinedInstructions,
            tools: tools + other.tools
        )
    }
}

private protocol PrimitiveDynamicInstructions {
    func resolve() -> ResolvedDynamicInstructions
}

private struct DynamicTool: DynamicInstructions, PrimitiveDynamicInstructions {
    typealias Body = Never

    let tool: any Tool

    init(_ tool: any Tool) {
        self.tool = tool
    }

    var body: Never {
        fatalError("DynamicTool has no body")
    }

    func resolve() -> ResolvedDynamicInstructions {
        ResolvedDynamicInstructions(instructions: nil, tools: [tool])
    }
}

extension AnyDynamicInstructions: PrimitiveDynamicInstructions {
    fileprivate func resolve() -> ResolvedDynamicInstructions {
        resolveValue()
    }
}

extension TupleDynamicInstructions: PrimitiveDynamicInstructions {
    fileprivate func resolve() -> ResolvedDynamicInstructions {
        var result = ResolvedDynamicInstructions.empty
        repeat result = result.appending(resolveDynamicInstructions(each contents))
        return result
    }
}

extension ConditionalDynamicInstructions: PrimitiveDynamicInstructions {
    fileprivate func resolve() -> ResolvedDynamicInstructions {
        switch branch {
        case .trueContent(let content):
            resolveDynamicInstructions(content)
        case .falseContent(let content):
            resolveDynamicInstructions(content)
        }
    }
}

extension Optional: PrimitiveDynamicInstructions where Wrapped: DynamicInstructions {
    fileprivate func resolve() -> ResolvedDynamicInstructions {
        map(resolveDynamicInstructions) ?? .empty
    }
}

extension Never: PrimitiveDynamicInstructions {
    fileprivate func resolve() -> ResolvedDynamicInstructions {
        switch self {}
    }
}

extension EmptyDynamicInstructions: PrimitiveDynamicInstructions {
    fileprivate func resolve() -> ResolvedDynamicInstructions {
        .empty
    }
}

extension DynamicInstructionsForEach: PrimitiveDynamicInstructions {
    fileprivate func resolve() -> ResolvedDynamicInstructions {
        data.reduce(into: .empty) { result, element in
            result = result.appending(resolveDynamicInstructions(content(element)))
        }
    }
}

extension Instructions: PrimitiveDynamicInstructions {
    fileprivate func resolve() -> ResolvedDynamicInstructions {
        ResolvedDynamicInstructions(instructions: self, tools: [])
    }
}

private func resolveDynamicInstructions(
    _ dynamicInstructions: any DynamicInstructions
) -> ResolvedDynamicInstructions {
    func resolve<Content>(_ content: Content) -> ResolvedDynamicInstructions
    where Content: DynamicInstructions {
        if let primitive = content as? any PrimitiveDynamicInstructions {
            return primitive.resolve()
        }
        return resolve(content.body)
    }

    return resolve(dynamicInstructions)
}
