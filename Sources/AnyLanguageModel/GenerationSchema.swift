import protocol Foundation.LocalizedError
import class Foundation.JSONEncoder
import class Foundation.JSONDecoder
import struct Foundation.Decimal

/// A type that describes the properties of an object and any guides
/// on their values.
///
/// Generation schemas guide the output of a ``SystemLanguageModel`` to deterministically
/// ensure the output is in the desired format.
public struct GenerationSchema: Equatable, Codable, CustomDebugStringConvertible, Sendable {
    indirect enum Node: Equatable, Codable, Sendable {
        case object(ObjectNode)
        case array(ArrayNode)
        case string(StringNode)
        case number(NumberNode)
        case boolean
        case null
        case anyOf([Node])
        case ref(String)

        // MARK: - Equatable

        static func == (lhs: GenerationSchema.Node, rhs: GenerationSchema.Node) -> Bool {
            switch (lhs, rhs) {
            case (.boolean, .boolean), (.null, .null):
                return true
            case (.ref(let lhsName), .ref(let rhsName)):
                return lhsName == rhsName
            case (.string(let lhsString), .string(let rhsString)):
                return lhsString.description == rhsString.description
                    && lhsString.pattern == rhsString.pattern
                    && lhsString.enumChoices == rhsString.enumChoices
            case (.number(let lhsNumber), .number(let rhsNumber)):
                return lhsNumber.description == rhsNumber.description
                    && lhsNumber.integerOnly == rhsNumber.integerOnly
                    && lhsNumber.minimum == rhsNumber.minimum
                    && lhsNumber.maximum == rhsNumber.maximum
            case (.array(let lhsArray), .array(let rhsArray)):
                return lhsArray.description == rhsArray.description
                    && lhsArray.minItems == rhsArray.minItems
                    && lhsArray.maxItems == rhsArray.maxItems
                    && lhsArray.items == rhsArray.items
            case (.object(let lhsObject), .object(let rhsObject)):
                return lhsObject.description == rhsObject.description
                    && lhsObject.required == rhsObject.required
                    && lhsObject.representsNilExplicitly == rhsObject.representsNilExplicitly
                    && (!lhsObject.representsNilExplicitly || lhsObject.propertyOrder == rhsObject.propertyOrder)
                    && lhsObject.properties.keys == rhsObject.properties.keys
                    && lhsObject.properties.allSatisfy { key, lhsNode in
                        guard let rhsNode = rhsObject.properties[key] else { return false }
                        return lhsNode == rhsNode
                    }
            case (.anyOf(let lhsNodes), .anyOf(let rhsNodes)):
                return lhsNodes.count == rhsNodes.count && zip(lhsNodes, rhsNodes).allSatisfy(==)
            default:
                return false
            }
        }

        // MARK: - Codable

        private enum CodingKeys: String, CodingKey {
            case type, properties, required, additionalProperties
            case items, minItems, maxItems
            case pattern, `enum`, anyOf
            case ref = "$ref"
            case description
            case minimum, maximum
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)

            switch self {
            case .object(let obj):
                try container.encode("object", forKey: .type)
                if let desc = obj.description {
                    try container.encode(desc, forKey: .description)
                }
                var propsContainer = container.nestedContainer(
                    keyedBy: GenerationSchema.DynamicCodingKey.self,
                    forKey: .properties
                )
                for name in obj.properties.keys.sorted() {
                    guard let node = obj.properties[name] else { continue }
                    try propsContainer.encode(node, forKey: GenerationSchema.DynamicCodingKey(stringValue: name)!)
                }
                try container.encode(obj.required.sorted(), forKey: .required)

                // Check userInfo to see if additionalProperties should be omitted
                let shouldOmit = encoder.userInfo[GenerationSchema.omitAdditionalPropertiesKey] as? Bool ?? false
                if !shouldOmit {
                    try container.encode(false, forKey: .additionalProperties)
                }

            case .array(let arr):
                try container.encode("array", forKey: .type)
                if let desc = arr.description {
                    try container.encode(desc, forKey: .description)
                }
                try container.encode(arr.items, forKey: .items)
                if let min = arr.minItems {
                    try container.encode(min, forKey: .minItems)
                }
                if let max = arr.maxItems {
                    try container.encode(max, forKey: .maxItems)
                }

            case .string(let str):
                try container.encode("string", forKey: .type)
                if let desc = str.description {
                    try container.encode(desc, forKey: .description)
                }
                if let pattern = str.pattern {
                    try container.encode(pattern, forKey: .pattern)
                }
                if let choices = str.enumChoices {
                    try container.encode(choices, forKey: .enum)
                }

            case .number(let num):
                try container.encode(num.integerOnly ? "integer" : "number", forKey: .type)
                if let desc = num.description {
                    try container.encode(desc, forKey: .description)
                }
                if let min = num.minimum {
                    try container.encode(min, forKey: .minimum)
                }
                if let max = num.maximum {
                    try container.encode(max, forKey: .maximum)
                }

            case .boolean:
                try container.encode("boolean", forKey: .type)

            case .null:
                try container.encode("null", forKey: .type)

            case .anyOf(let nodes):
                try container.encode(nodes, forKey: .anyOf)

            case .ref(let name):
                try container.encode("#/$defs/\(name)", forKey: .ref)
            }
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)

            if container.contains(.ref) {
                let refString = try container.decode(String.self, forKey: .ref)
                let name = refString.replacingOccurrences(of: "#/$defs/", with: "")
                self = .ref(name)
                return
            }

            if container.contains(.anyOf) {
                let nodes = try container.decode([GenerationSchema.Node].self, forKey: .anyOf)
                self = .anyOf(nodes)
                return
            }

            let type = try container.decode(String.self, forKey: .type)
            let description = try container.decodeIfPresent(String.self, forKey: .description)

            switch type {
            case "object":
                let propsContainer = try container.nestedContainer(
                    keyedBy: GenerationSchema.DynamicCodingKey.self,
                    forKey: .properties
                )
                var properties: [String: GenerationSchema.Node] = [:]
                for key in propsContainer.allKeys {
                    properties[key.stringValue] = try propsContainer.decode(GenerationSchema.Node.self, forKey: key)
                }
                let requiredArray = try container.decodeIfPresent([String].self, forKey: .required) ?? []
                let required = Set(requiredArray)
                self = .object(
                    GenerationSchema.ObjectNode(description: description, properties: properties, required: required)
                )

            case "array":
                let items = try container.decode(GenerationSchema.Node.self, forKey: .items)
                let minItems = try container.decodeIfPresent(Int.self, forKey: .minItems)
                let maxItems = try container.decodeIfPresent(Int.self, forKey: .maxItems)
                self = .array(
                    GenerationSchema.ArrayNode(
                        description: description,
                        items: items,
                        minItems: minItems,
                        maxItems: maxItems
                    )
                )

            case "string":
                let pattern = try container.decodeIfPresent(String.self, forKey: .pattern)
                let enumChoices = try container.decodeIfPresent([String].self, forKey: .enum)
                self = .string(
                    GenerationSchema.StringNode(description: description, pattern: pattern, enumChoices: enumChoices)
                )

            case "number", "integer":
                let minimum = try container.decodeIfPresent(Double.self, forKey: .minimum)
                let maximum = try container.decodeIfPresent(Double.self, forKey: .maximum)
                self = .number(
                    GenerationSchema.NumberNode(
                        description: description,
                        minimum: minimum,
                        maximum: maximum,
                        integerOnly: type == "integer"
                    )
                )

            case "boolean":
                self = .boolean

            case "null":
                self = .null

            default:
                throw DecodingError.dataCorruptedError(
                    forKey: .type,
                    in: container,
                    debugDescription: "Unknown type: \(type)"
                )
            }
        }
    }

    struct ObjectNode: Sendable, Codable {
        var description: String?
        var properties: [String: Node]
        var required: Set<String>
        /// Whether generated content has a `null` value for each optional property
        /// that it would otherwise leave out.
        var representsNilExplicitly = false
        /// The property names in declaration order, when known.
        var propertyOrder: [String] = []

        private enum CodingKeys: String, CodingKey {
            case description, properties, required
        }
    }

    struct ArrayNode: Sendable, Codable {
        var description: String?
        var items: Node
        var minItems: Int?
        var maxItems: Int?
    }

    struct StringNode: Sendable, Codable {
        var description: String?
        var pattern: String?
        var enumChoices: [String]?
    }

    struct NumberNode: Sendable, Codable {
        var description: String?
        var minimum: Double?
        var maximum: Double?
        var integerOnly: Bool
    }

    let root: Node
    var defs: [String: Node]

    /// A string representation of the debug description.
    ///
    /// This string is not localized and is not appropriate for display to end users.
    public var debugDescription: String {
        var parts: [String] = []
        parts.append("GenerationSchema:")
        parts.append("  root: \(debugString(for: root, indent: 2))")
        if !defs.isEmpty {
            parts.append("  $defs:")
            for (name, node) in defs.sorted(by: { $0.key < $1.key }) {
                parts.append("    \(name): \(debugString(for: node, indent: 4))")
            }
        }
        return parts.joined(separator: "\n")
    }

    private func debugString(for node: Node, indent: Int) -> String {
        switch node {
        case .object(let obj):
            return "object(\(obj.properties.count) properties)"
        case .array(let arr):
            return "array(items: \(debugString(for: arr.items, indent: 0)))"
        case .string(let str):
            if let choices = str.enumChoices {
                return "string(enum: \(choices))"
            } else if str.pattern != nil {
                return "string(pattern)"
            }
            return "string"
        case .number(let num):
            return num.integerOnly ? "integer" : "number"
        case .boolean:
            return "boolean"
        case .null:
            return "null"
        case .anyOf(let nodes):
            return "anyOf(\(nodes.count) choices)"
        case .ref(let name):
            return "$ref(\(name))"
        }
    }

    /// Creates a schema by providing an array of properties.
    ///
    /// - Parameters:
    ///   - type: The type this schema represents.
    ///   - description: A natural language description of this schema.
    ///   - properties: An array of properties.
    public init(
        type: any Generable.Type,
        description: String? = nil,
        properties: [GenerationSchema.Property]
    ) {
        self.init(type: type, description: description, explicitNil: false, properties: properties)
    }

    /// Creates a schema by providing an array of properties.
    ///
    /// - Parameters:
    ///   - type: The type this schema represents.
    ///   - description: A natural language description of this schema.
    ///   - explicitNil: Whether generated content has a `null` value
    ///     for each optional property that it would otherwise leave out.
    ///     Like Foundation Models,
    ///     the schema's encoded form doesn't include this setting.
    ///   - properties: An array of properties.
    public init(
        type: any Generable.Type,
        description: String? = nil,
        representNilExplicitlyInGeneratedContent explicitNil: Bool,
        properties: [GenerationSchema.Property]
    ) {
        self.init(type: type, description: description, explicitNil: explicitNil, properties: properties)
    }

    private init(
        type: any Generable.Type,
        description: String?,
        explicitNil: Bool,
        properties: [GenerationSchema.Property]
    ) {
        let typeName = String(reflecting: type)
        var props: [String: Node] = [:]
        var required: Set<String> = []
        var allDefs: [String: Node] = [:]

        for property in properties {
            props[property.name] = property.node
            if !property.isOptional {
                required.insert(property.name)
            }
            for (defName, defNode) in property.deps {
                if let existing = allDefs[defName], existing != defNode {
                    fatalError("Duplicate type '\(defName)' with different structure")
                }
                allDefs[defName] = defNode
            }
        }

        let objectNode = ObjectNode(
            description: description,
            properties: props,
            required: required,
            representsNilExplicitly: explicitNil,
            propertyOrder: properties.map(\.name)
        )
        allDefs[typeName] = .object(objectNode)

        self.root = .ref(typeName)
        self.defs = allDefs
    }

    /// Creates a schema for a string enumeration.
    ///
    /// - Parameters:
    ///   - type: The type this schema represents.
    ///   - description: A natural language description of this schema.
    ///   - anyOf: The allowed choices.
    public init(
        type: any Generable.Type,
        description: String? = nil,
        anyOf choices: [String]
    ) {
        guard !choices.isEmpty else {
            fatalError("Empty choices for enum schema")
        }
        let node = StringNode(description: description, pattern: nil, enumChoices: choices)
        self.root = .string(node)
        self.defs = [:]
    }

    /// Creates a schema as the union of several other types.
    ///
    /// - Parameters:
    ///   - type: The type this schema represents.
    ///   - description: A natural language description of this schema.
    ///   - anyOf: The types this schema should be a union of.
    public init(
        type: any Generable.Type,
        description: String? = nil,
        anyOf types: [any Generable.Type]
    ) {
        guard !types.isEmpty else {
            fatalError("Empty types for anyOf schema")
        }

        var members: [Node] = []
        var allDefs: [String: Node] = [:]

        for t in types {
            let tName = String(reflecting: t)
            members.append(.ref(tName))

            let tSchema = t.generationSchema
            for (defName, defNode) in tSchema.defs {
                if let existing = allDefs[defName], existing != defNode {
                    fatalError("Duplicate type '\(defName)' with different structure")
                }
                allDefs[defName] = defNode
            }

            if case .ref(_) = tSchema.root {
                // Already in defs
            } else {
                allDefs[tName] = tSchema.root
            }
        }

        self.root = .anyOf(members)
        self.defs = allDefs
    }

    /// Creates a schema by providing an array of dynamic schemas.
    ///
    /// - Parameters:
    ///   - root: The root schema.
    ///   - dependencies: An array of dynamic schemas.
    /// - Throws: Throws there are schemas with naming conflicts or
    ///   references to undefined types.
    public init(root: DynamicGenerationSchema, dependencies: [DynamicGenerationSchema]) throws {
        var nameMap: [String: DynamicGenerationSchema] = [:]
        var allDefs: [String: Node] = [:]

        // Build name map
        for dep in dependencies {
            if let name = dep.name {
                if nameMap[name] != nil {
                    throw SchemaError.duplicateType(
                        schema: nil,
                        type: name,
                        context: SchemaError.Context(debugDescription: "Duplicate dependency name")
                    )
                }
                nameMap[name] = dep
            }
        }

        if let rootName = root.name {
            if nameMap[rootName] != nil {
                throw SchemaError.duplicateType(
                    schema: nil,
                    type: rootName,
                    context: SchemaError.Context(debugDescription: "Root name conflicts with dependency")
                )
            }
            nameMap[rootName] = root
        }

        // Convert root
        let rootNode = try Self.convertDynamic(root, nameMap: nameMap, defs: &allDefs)

        // Convert all dependencies
        for dep in dependencies {
            _ = try Self.convertDynamic(dep, nameMap: nameMap, defs: &allDefs)
        }

        // Validate all references
        var undefinedRefs: [String] = []
        try Self.validateRefs(rootNode, defs: allDefs, undefinedRefs: &undefinedRefs)
        for (_, defNode) in allDefs {
            try Self.validateRefs(defNode, defs: allDefs, undefinedRefs: &undefinedRefs)
        }

        if !undefinedRefs.isEmpty {
            throw SchemaError.undefinedReferences(
                schema: root.name,
                references: Array(Set(undefinedRefs)),
                context: SchemaError.Context(debugDescription: "Undefined references")
            )
        }

        self.root = rootNode
        self.defs = allDefs
    }

    private init(root: Node, defs: [String: Node]) {
        self.root = root
        self.defs = defs
    }

    static func primitive<T: Generable>(_: T.Type, node: Node) -> GenerationSchema {
        GenerationSchema(root: node, defs: [:])
    }

    func withResolvedRoot() -> GenerationSchema? {
        if case .ref(let refName) = root,
            let defNode = defs[refName]
        {
            return GenerationSchema(root: defNode, defs: defs)
        }
        return nil
    }

    private static func convertDynamic(
        _ dynamic: DynamicGenerationSchema,
        nameMap: [String: DynamicGenerationSchema],
        defs: inout [String: Node],
        dynamicProp: DynamicGenerationSchema.Property? = nil
    ) throws -> Node {
        switch dynamic.body {
        case .object(let name, let desc, let properties):
            var props: [String: Node] = [:]
            var required: Set<String> = []
            for prop in properties {
                props[prop.name] = try convertDynamic(prop.schema, nameMap: nameMap, defs: &defs, dynamicProp: prop)
                if !prop.isOptional {
                    required.insert(prop.name)
                }
            }
            let node = Node.object(
                ObjectNode(
                    description: desc,
                    properties: props,
                    required: required,
                    representsNilExplicitly: dynamic.representsNilExplicitly,
                    propertyOrder: properties.map(\.name)
                )
            )
            if let name = name {
                defs[name] = node
                return .ref(name)
            }
            return node

        case .anyOf(let name, _, let choices):
            let nodes = try choices.map { try convertDynamic($0, nameMap: nameMap, defs: &defs) }
            let node = Node.anyOf(nodes)
            if let name = name {
                defs[name] = node
                return .ref(name)
            }
            return node

        case .stringEnum(let name, let desc, let choices):
            guard !choices.isEmpty else {
                throw SchemaError.emptyTypeChoices(
                    schema: name ?? "",
                    context: SchemaError.Context(debugDescription: "Empty enum choices")
                )
            }
            let node = Node.string(StringNode(description: desc, pattern: nil, enumChoices: choices))
            if let name = name {
                defs[name] = node
                return .ref(name)
            }
            return node

        case .array(let item, let min, let max):
            let itemNode = try convertDynamic(item, nameMap: nameMap, defs: &defs)
            return .array(
                ArrayNode(description: dynamicProp?.description, items: itemNode, minItems: min, maxItems: max)
            )

        case .scalar(let scalar):
            switch scalar {
            case .bool:
                return .boolean
            case .null:
                return .null
            case .string:
                return .string(StringNode(description: dynamicProp?.description, pattern: nil, enumChoices: nil))
            case .number:
                return .number(
                    NumberNode(description: dynamicProp?.description, minimum: nil, maximum: nil, integerOnly: false)
                )
            case .integer:
                return .number(
                    NumberNode(description: dynamicProp?.description, minimum: nil, maximum: nil, integerOnly: true)
                )
            case .decimal:
                return .number(
                    NumberNode(description: dynamicProp?.description, minimum: nil, maximum: nil, integerOnly: false)
                )
            }

        case .reference(let name):
            return .ref(name)
        }
    }

    private static func validateRefs(_ node: Node, defs: [String: Node], undefinedRefs: inout [String]) throws {
        switch node {
        case .ref(let name):
            if defs[name] == nil {
                undefinedRefs.append(name)
            }
        case .object(let obj):
            for (_, propNode) in obj.properties {
                try validateRefs(propNode, defs: defs, undefinedRefs: &undefinedRefs)
            }
        case .array(let arr):
            try validateRefs(arr.items, defs: defs, undefinedRefs: &undefinedRefs)
        case .anyOf(let nodes):
            for n in nodes {
                try validateRefs(n, defs: defs, undefinedRefs: &undefinedRefs)
            }
        default:
            break
        }
    }

    // MARK: - Codable

    private enum CodingKeys: String, CodingKey {
        case defs = "$defs"
        case ref = "$ref"
    }

    private struct DynamicCodingKey: CodingKey {
        var stringValue: String
        var intValue: Int?

        init?(stringValue: String) {
            self.stringValue = stringValue
            self.intValue = nil
        }

        init?(intValue: Int) {
            self.stringValue = String(intValue)
            self.intValue = intValue
        }
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        if container.contains(.defs) {
            let defsContainer = try container.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .defs)
            var defs: [String: Node] = [:]
            for key in defsContainer.allKeys {
                defs[key.stringValue] = try defsContainer.decode(Node.self, forKey: key)
            }
            self.defs = defs
        } else {
            self.defs = [:]
        }

        // Decode the root - could be inline or a ref
        if container.contains(.ref) {
            let refString = try container.decode(String.self, forKey: .ref)
            let name = refString.replacingOccurrences(of: "#/$defs/", with: "")
            self.root = .ref(name)
        } else {
            // Inline root - decode as a node
            self.root = try Node(from: decoder)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        if !defs.isEmpty {
            var defsContainer = container.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .defs)
            for (name, node) in defs.sorted(by: { $0.key < $1.key }) {
                try defsContainer.encode(node, forKey: DynamicCodingKey(stringValue: name)!)
            }
        }

        // Encode root
        if case .ref(_) = root {
            try root.encode(to: encoder)
        } else {
            try root.encode(to: encoder)
        }
    }
}

// MARK: - GenerationSchema.Property

extension GenerationSchema {
    /// A property that belongs to a generation schema.
    ///
    /// Fields are named members of object types. Fields are strongly
    /// typed and have optional descriptions and guides.
    public struct Property: Sendable {
        let name: String
        let node: Node
        let isOptional: Bool
        var deps: [String: Node]

        /// Create a property that contains a generable type.
        ///
        /// - Parameters:
        ///   - name: The property's name.
        ///   - description: A natural language description of what content
        ///     should be generated for this property.
        ///   - type: The type this property represents.
        ///   - guides: A list of guides to apply to this property.
        public init<Value>(
            name: String,
            description: String? = nil,
            type: Value.Type,
            guides: [GenerationGuide<Value>] = []
        ) where Value: Generable {
            self.name = name
            self.isOptional = false

            let (node, deps) = Self.buildNode(for: Value.self, description: description, guides: guides)
            self.node = node
            self.deps = deps
        }

        /// Create an optional property that contains a generable type.
        ///
        /// - Parameters:
        ///   - name: The property's name.
        ///   - description: A natural language description of what content
        ///     should be generated for this property.
        ///   - type: The type this property represents.
        ///   - guides: A list of guides to apply to this property.
        public init<Value>(
            name: String,
            description: String? = nil,
            type: Value?.Type,
            guides: [GenerationGuide<Value>] = []
        ) where Value: Generable {
            self.name = name
            self.isOptional = true

            let (node, deps) = Self.buildNode(for: Value.self, description: description, guides: guides)
            self.node = node
            self.deps = deps
        }

        /// Create a property that contains a string type.
        ///
        /// - Parameters:
        ///   - name: The property's name.
        ///   - description: A natural language description of what content
        ///     should be generated for this property.
        ///   - type: The type this property represents.
        ///   - guides: An array of regexes to be applied to this string. If there're multiple regexes in the array, only the last one will be applied.
        public init<RegexOutput>(
            name: String,
            description: String? = nil,
            type: String.Type,
            guides: [Regex<RegexOutput>] = []
        ) {
            self.name = name
            self.isOptional = false
            self.node = .string(StringNode(description: description, pattern: nil, enumChoices: nil))
            self.deps = [:]
        }

        /// Create an optional property that contains a generable type.
        ///
        /// - Parameters:
        ///   - name: The property's name.
        ///   - description: A natural language description of what content
        ///     should be generated for this property.
        ///   - type: The type this property represents.
        ///   - guides: An array of regexes to be applied to this string. If there're multiple regexes in the array, only the last one will be applied.
        public init<RegexOutput>(
            name: String,
            description: String? = nil,
            type: String?.Type,
            guides: [Regex<RegexOutput>] = []
        ) {
            self.name = name
            self.isOptional = true
            self.node = .string(StringNode(description: description, pattern: nil, enumChoices: nil))
            self.deps = [:]
        }

        private static func buildNode<Value: Generable>(
            for type: Value.Type,
            description: String?,
            guides: [GenerationGuide<Value>]
        ) -> (Node, [String: Node]) {
            // Check if it's a primitive type
            if type == Bool.self {
                return (.boolean, [:])
            } else if type == String.self {
                return (.string(StringNode(description: description, pattern: nil, enumChoices: nil)), [:])
            } else if type == Int.self {
                var minimum: Double?
                var maximum: Double?
                for guide in guides {
                    if let min = guide.minimum { minimum = min }
                    if let max = guide.maximum { maximum = max }
                }
                return (
                    .number(
                        NumberNode(description: description, minimum: minimum, maximum: maximum, integerOnly: true)
                    ), [:]
                )
            } else if type == Float.self || type == Double.self || type == Decimal.self {
                var minimum: Double?
                var maximum: Double?
                for guide in guides {
                    if let min = guide.minimum { minimum = min }
                    if let max = guide.maximum { maximum = max }
                }
                return (
                    .number(
                        NumberNode(description: description, minimum: minimum, maximum: maximum, integerOnly: false)
                    ), [:]
                )
            } else {
                // Complex type - use its schema
                let schema = Value.generationSchema

                // Arrays should be inlined, not referenced
                if case .array(let arrayNode) = schema.root {
                    var updatedArrayNode = arrayNode
                    updatedArrayNode.description = description
                    for guide in guides {
                        if let min = guide.minimumCount { updatedArrayNode.minItems = min }
                        if let max = guide.maximumCount { updatedArrayNode.maxItems = max }
                    }
                    return (.array(updatedArrayNode), schema.defs)
                }

                let typeName = String(reflecting: Value.self)

                var deps = schema.defs
                if case .ref(_) = schema.root {
                    // Already a ref
                } else {
                    deps[typeName] = schema.root
                }

                return (.ref(typeName), deps)
            }
        }
    }
}

// MARK: - GenerationSchema.SchemaError

extension GenerationSchema {
    /// A error that occurs when there is a problem creating a generation schema.
    public enum SchemaError: Error, LocalizedError {

        /// The context in which the error occurred.
        public struct Context: Sendable {

            /// A string representation of the debug description.
            ///
            /// This string is not localized and is not appropriate for display to end users.
            public let debugDescription: String

            public init(debugDescription: String) {
                self.debugDescription = debugDescription
            }
        }

        /// An error that represents an attempt to construct a schema from dynamic schemas,
        /// and two or more of the subschemas have the same type name.
        case duplicateType(schema: String?, type: String, context: Context)

        /// An error that represents an attempt to construct a dynamic schema
        /// with properties that have conflicting names.
        case duplicateProperty(schema: String, property: String, context: Context)

        /// An error that represents an attempt to construct an anyOf schema with an
        /// empty array of type choices.
        case emptyTypeChoices(schema: String, context: Context)

        /// An error that represents an attempt to construct a schema from dynamic schemas,
        /// and one of those schemas references an undefined schema.
        case undefinedReferences(schema: String?, references: [String], context: Context)

        /// A string representation of the error description.
        public var errorDescription: String? {
            switch self {
            case .duplicateType(let schema, let type, _):
                return "Duplicate type '\(type)' in schema '\(schema ?? "root")'"
            case .duplicateProperty(let schema, let property, _):
                return "Duplicate property '\(property)' in schema '\(schema)'"
            case .emptyTypeChoices(let schema, _):
                return "Empty type choices in schema '\(schema)'"
            case .undefinedReferences(let schema, let references, _):
                return "Undefined references \(references) in schema '\(schema ?? "root")'"
            }
        }

        /// A suggestion that indicates how to handle the error.
        public var recoverySuggestion: String? {
            switch self {
            case .duplicateType:
                return "Ensure all types have unique names"
            case .duplicateProperty:
                return "Ensure all properties have unique names"
            case .emptyTypeChoices:
                return "Provide at least one type choice"
            case .undefinedReferences:
                return "Ensure all referenced schemas are defined"
            }
        }
    }
}

// MARK: - CodingUserInfoKey

extension GenerationSchema {
    /// A key used in the encoder's `userInfo` dictionary to control whether
    /// the `additionalProperties` field should be omitted from the encoded output.
    ///
    /// Set this to `true` to omit `additionalProperties` from object schemas.
    /// Defaults to `false` (includes `additionalProperties`) if not specified.
    ///
    /// Example:
    /// ```swift
    /// let encoder = JSONEncoder()
    /// encoder.userInfo[GenerationSchema.omitAdditionalPropertiesKey] = true
    /// let data = try encoder.encode(schema)
    /// ```
    static let omitAdditionalPropertiesKey = CodingUserInfoKey(rawValue: "GenerationSchema.omitAdditionalProperties")!

    func schemaPrompt() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(self),
            let schemaJSON = String(data: data, encoding: .utf8)
        else {
            return "Respond with valid JSON only."
        }
        return "Respond with valid JSON matching this schema:\n\(schemaJSON)"
    }
}

// MARK: - Explicit nil

extension GenerationSchema {
    /// Returns generated content with a `null` value for each optional property
    /// that the content leaves out,
    /// in objects whose schema represents `nil` explicitly.
    ///
    /// Content for a schema without such objects is returned unchanged.
    func representingNilExplicitly(in content: GeneratedContent) -> GeneratedContent {
        let representsNilExplicitly = ([root] + Array(defs.values)).contains { node in
            if case .object(let object) = node { return object.representsNilExplicitly }
            return false
        }
        guard representsNilExplicitly else { return content }
        return representingNilExplicitly(in: content, node: root, depth: 0)
    }

    private func representingNilExplicitly(
        in content: GeneratedContent,
        node: Node,
        depth: Int
    ) -> GeneratedContent {
        guard depth < 64 else { return content }
        switch node {
        case .ref(let name):
            guard let resolved = defs[name] else { return content }
            return representingNilExplicitly(in: content, node: resolved, depth: depth + 1)
        case .object(let object):
            guard case .structure(var properties, var orderedKeys) = content.kind else { return content }
            for (key, value) in properties {
                if let child = object.properties[key] {
                    properties[key] = representingNilExplicitly(in: value, node: child, depth: depth + 1)
                }
            }
            if object.representsNilExplicitly {
                let declaredKeys = object.propertyOrder.isEmpty ? object.properties.keys.sorted() : object.propertyOrder
                for key in declaredKeys
                where properties[key] == nil && !object.required.contains(key) {
                    properties[key] = GeneratedContent(kind: .null)
                    orderedKeys.append(key)
                }
                // Put declared properties in declaration order, followed by any others.
                if !object.propertyOrder.isEmpty {
                    let declared = Set(object.propertyOrder)
                    orderedKeys =
                        object.propertyOrder.filter { properties[$0] != nil }
                        + orderedKeys.filter { !declared.contains($0) }
                }
            }
            return GeneratedContent(kind: .structure(properties: properties, orderedKeys: orderedKeys), id: content.id)
        case .array(let array):
            guard case .array(let elements) = content.kind else { return content }
            let items = elements.map { representingNilExplicitly(in: $0, node: array.items, depth: depth + 1) }
            return GeneratedContent(kind: .array(items), id: content.id)
        case .anyOf(let variants):
            guard let variant = variant(matching: content, among: variants, depth: depth) else { return content }
            return representingNilExplicitly(in: content, node: variant, depth: depth + 1)
        case .string, .number, .boolean, .null:
            return content
        }
    }

    /// Returns the first variant whose shape matches the content:
    /// an object that declares every property in a structure
    /// and whose required properties the structure has,
    /// an array for an array,
    /// or a nested union with a matching variant.
    private func variant(matching content: GeneratedContent, among variants: [Node], depth: Int) -> Node? {
        variants.first { variant in
            switch (resolving(variant, depth: depth), content.kind) {
            case (.object(let object)?, .structure(let properties, _)):
                return properties.keys.allSatisfy { object.properties[$0] != nil }
                    && object.required.allSatisfy { properties[$0] != nil }
            case (.array?, .array):
                return true
            case (.anyOf(let nested)?, _):
                guard depth < 64 else { return false }
                return self.variant(matching: content, among: nested, depth: depth + 1) != nil
            default:
                return false
            }
        }
    }

    private func resolving(_ node: Node, depth: Int) -> Node? {
        guard depth < 64 else { return nil }
        guard case .ref(let name) = node else { return node }
        return defs[name].flatMap { resolving($0, depth: depth + 1) }
    }
}
