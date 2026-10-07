import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

/// Conforms a type to ``Generable`` protocol.
public struct GenerableMacro: MemberMacro, ExtensionMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingMembersOf declaration: some DeclGroupSyntax,
        conformingTo protocols: [TypeSyntax],
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        if let structDecl = declaration.as(StructDeclSyntax.self) {
            let structName = structDecl.name.text

            let description = extractDescription(from: node)
            let explicitNil = try extractRepresentNilExplicitly(from: node)
            let properties = extractGuidedProperties(from: structDecl)

            return [
                generateRawContentProperty(),
                generateMemberwiseInit(properties: properties, explicitNil: explicitNil),
                generateInitFromGeneratedContent(structName: structName, properties: properties),
                generateGeneratedContentProperty(
                    structName: structName,
                    description: description,
                    properties: properties,
                    explicitNil: explicitNil
                ),
                generateGenerationSchemaProperty(
                    structName: structName,
                    description: description,
                    properties: properties,
                    explicitNil: explicitNil
                ),
                generatePartiallyGeneratedStruct(structName: structName, properties: properties),
                generateAsPartiallyGeneratedMethod(structName: structName),
                generateInstructionsRepresentationProperty(),
                generatePromptRepresentationProperty(),
            ]
        } else if let enumDecl = declaration.as(EnumDeclSyntax.self) {
            let enumName = enumDecl.name.text

            let description = extractDescription(from: node)
            let cases = extractEnumCases(from: enumDecl)

            return [
                generateEnumInitFromGeneratedContent(enumName: enumName, cases: cases),
                generateEnumGeneratedContentProperty(
                    enumName: enumName,
                    description: description,
                    cases: cases
                ),
                generateEnumGenerationSchemaProperty(
                    enumName: enumName,
                    description: description,
                    cases: cases
                ),
                generateAsPartiallyGeneratedMethodForEnum(enumName: enumName),
                generateInstructionsRepresentationProperty(),
                generatePromptRepresentationProperty(),
            ]
        } else {
            throw GenerableMacroError.notApplicableToType
        }
    }

    public static func expansion(
        of node: AttributeSyntax,
        attachedTo declaration: some DeclGroupSyntax,
        providingExtensionsOf type: some TypeSyntaxProtocol,
        conformingTo protocols: [TypeSyntax],
        in context: some MacroExpansionContext
    ) throws -> [ExtensionDeclSyntax] {
        let nonisolatedModifier = DeclModifierSyntax(name: .keyword(.nonisolated))

        let extensionDecl = ExtensionDeclSyntax(
            modifiers: DeclModifierListSyntax([nonisolatedModifier]),
            extendedType: type,
            inheritanceClause: InheritanceClauseSyntax(
                inheritedTypes: InheritedTypeListSyntax([
                    InheritedTypeSyntax(type: TypeSyntax(stringLiteral: "Generable"))
                ])
            ),
            memberBlock: MemberBlockSyntax(members: [])
        )
        return [extensionDecl]
    }

    // MARK: - Helpers

    /// Returns whether the attribute passes `representNilExplicitlyInGeneratedContent: true`.
    ///
    /// The macro reads the value when it expands the type,
    /// so the argument must be a Boolean literal.
    private static func extractRepresentNilExplicitly(from node: AttributeSyntax) throws -> Bool {
        guard let arguments = node.arguments?.as(LabeledExprListSyntax.self),
            let argument = arguments.first(where: { $0.label?.text == "representNilExplicitlyInGeneratedContent" })
        else {
            return false
        }
        guard let literal = argument.expression.as(BooleanLiteralExprSyntax.self) else {
            throw GenerableMacroError.nonLiteralRepresentNilExplicitly
        }
        return literal.literal.tokenKind == .keyword(.true)
    }

    /// Returns a statement that adds an optional property to `properties`.
    ///
    /// When `explicitNil` is `true`, a `nil` value becomes `null`;
    /// otherwise, the property is left out.
    private static func optionalPropertyConversion(
        propertyName propName: String,
        baseType: String,
        explicitNil: Bool
    ) -> String {
        let conversion: String
        if baseType == "String" {
            conversion = "GeneratedContent(value)"
        } else if baseType.hasPrefix("[") && baseType.hasSuffix("]") && !isDictionaryType(baseType) {
            conversion = "GeneratedContent(elements: value)"
        } else {
            conversion = "value.generatedContent"
        }
        if explicitNil {
            return """
                if let value = \(propName) {
                            properties["\(propName)"] = \(conversion)
                        } else {
                            properties["\(propName)"] = GeneratedContent(kind: .null)
                        }
                """
        }
        return """
            if let value = \(propName) {
                        properties["\(propName)"] = \(conversion)
                    }
            """
    }

    private static func extractDescription(from node: AttributeSyntax) -> String? {
        guard let arguments = node.arguments?.as(LabeledExprListSyntax.self),
            let firstArg = arguments.first,
            firstArg.label?.text == "description",
            let stringLiteral = firstArg.expression.as(StringLiteralExprSyntax.self)
        else {
            return nil
        }
        return stringLiteral.segments.description.trimmingCharacters(in: .init(charactersIn: "\""))
    }

    private static func extractGuidedProperties(from structDecl: StructDeclSyntax) -> [PropertyInfo] {
        var properties: [PropertyInfo] = []

        for member in structDecl.memberBlock.members {
            if let varDecl = member.decl.as(VariableDeclSyntax.self),
                let binding = varDecl.bindings.first,
                let identifier = binding.pattern.as(IdentifierPatternSyntax.self)
            {
                let propertyName = identifier.identifier.text
                let propertyType = binding.typeAnnotation?.type.description ?? "String"
                let guideInfo = extractGuideInfo(from: varDecl.attributes)

                properties.append(
                    PropertyInfo(
                        name: propertyName,
                        type: propertyType,
                        guide: guideInfo
                    )
                )
            }
        }

        return properties
    }

    private static func extractGuideInfo(from attributes: AttributeListSyntax) -> GuideInfo {
        for attribute in attributes {
            if let attr = attribute.as(AttributeSyntax.self),
                attr.attributeName.description == "Guide"
            {
                if let arguments = attr.arguments?.as(LabeledExprListSyntax.self) {
                    var description: String?
                    var constraints = Constraints()

                    for arg in arguments {
                        if arg.label?.text == "description",
                            let stringLiteral = arg.expression.as(StringLiteralExprSyntax.self)
                        {
                            description = stringLiteral.segments.description.trimmingCharacters(
                                in: .init(charactersIn: "\"")
                            )
                            continue
                        }

                        if description == nil,
                            arg.label == nil,
                            let stringLiteral = arg.expression.as(StringLiteralExprSyntax.self)
                        {
                            description = stringLiteral.segments.description.trimmingCharacters(
                                in: .init(charactersIn: "\"")
                            )
                            continue
                        }

                        let guideExpression = arg.expression
                        if let parsedPattern = parsePatternFromExpression(guideExpression) {
                            constraints.pattern = parsedPattern
                            continue
                        }

                        if isStringChoiceGuide(guideExpression) {
                            constraints.stringGuides.append(guideExpression.trimmedDescription)
                            continue
                        }

                        if let functionCall = guideExpression.as(FunctionCallExprSyntax.self) {
                            applyConstraints(from: functionCall, into: &constraints)
                        } else if let memberAccess = guideExpression.as(MemberAccessExprSyntax.self),
                            let functionCall = memberAccess.base?.as(FunctionCallExprSyntax.self)
                        {
                            applyConstraints(from: functionCall, into: &constraints)
                        }
                    }

                    return GuideInfo(description: description, constraints: constraints)
                }
            }
        }
        return GuideInfo(description: nil, constraints: Constraints())
    }

    private static func applyConstraints(from call: FunctionCallExprSyntax, into constraints: inout Constraints) {
        let functionName: String?
        if let memberAccess = call.calledExpression.as(MemberAccessExprSyntax.self) {
            functionName = memberAccess.declName.baseName.text
        } else if let identifier = call.calledExpression.as(DeclReferenceExprSyntax.self) {
            functionName = identifier.baseName.text
        } else {
            functionName = nil
        }

        guard let functionName, let firstArgument = call.arguments.first else { return }

        switch functionName {
        case "count":
            if let intLiteral = firstArgument.expression.as(IntegerLiteralExprSyntax.self),
                let value = Int(intLiteral.literal.text)
            {
                constraints.minimumCount = value
                constraints.maximumCount = value
            } else if let rangeExpression = firstArgument.expression.as(SequenceExprSyntax.self) {
                let (minimum, maximum) = parseClosedRangeInt(rangeExpression)
                constraints.minimumCount = minimum
                constraints.maximumCount = maximum
            }
        case "minimumCount":
            if let intLiteral = firstArgument.expression.as(IntegerLiteralExprSyntax.self),
                let value = Int(intLiteral.literal.text)
            {
                constraints.minimumCount = value
            }
        case "maximumCount":
            if let intLiteral = firstArgument.expression.as(IntegerLiteralExprSyntax.self),
                let value = Int(intLiteral.literal.text)
            {
                constraints.maximumCount = value
            }
        case "minimum":
            constraints.minimum = parseNumericLiteral(firstArgument.expression)
        case "maximum":
            constraints.maximum = parseNumericLiteral(firstArgument.expression)
        case "range":
            if let rangeExpression = firstArgument.expression.as(SequenceExprSyntax.self) {
                let (minimum, maximum) = parseClosedRangeDouble(rangeExpression)
                constraints.minimum = minimum
                constraints.maximum = maximum
            }
        default:
            break
        }
    }

    /// Whether `expression` is a `.constant(_:)` or `.anyOf(_:)` string guide,
    /// which is passed through to the generated schema as written.
    private static func isStringChoiceGuide(_ expression: ExprSyntax) -> Bool {
        guard let functionCall = expression.as(FunctionCallExprSyntax.self),
            let memberAccess = functionCall.calledExpression.as(MemberAccessExprSyntax.self)
        else { return false }
        let name = memberAccess.declName.baseName.text
        return name == "constant" || name == "anyOf"
    }

    private static func parsePatternFromExpression(_ expression: ExprSyntax) -> String? {
        if let functionCall = expression.as(FunctionCallExprSyntax.self) {
            let functionName: String?
            if let memberAccess = functionCall.calledExpression.as(MemberAccessExprSyntax.self) {
                functionName = memberAccess.declName.baseName.text
            } else if let identifier = functionCall.calledExpression.as(DeclReferenceExprSyntax.self) {
                functionName = identifier.baseName.text
            } else {
                functionName = nil
            }

            if functionName == "pattern",
                let firstArg = functionCall.arguments.first,
                let stringLiteral = firstArg.expression.as(StringLiteralExprSyntax.self)
            {
                return stringLiteral.segments.description.trimmingCharacters(in: .init(charactersIn: "\""))
            }
        }
        return nil
    }

    private static func topLevelColonIndex(in text: String) -> String.Index? {
        var totalDepth = 0

        for index in text.indices {
            switch text[index] {
            case "[":
                totalDepth += 1
            case "]":
                totalDepth -= 1
            case "<":
                totalDepth += 1
            case ">":
                totalDepth -= 1
            case "(":
                totalDepth += 1
            case ")":
                totalDepth -= 1
            case ":" where totalDepth == 0:
                return index
            default:
                break
            }

            if totalDepth < 0 {
                return nil
            }
        }

        return nil
    }

    private static func escapeDescriptionString(_ description: String?) -> String {
        guard let description else { return "nil" }
        return makeSwiftStringLiteralExpression(description)
    }

    /// Escapes text so it can be embedded safely inside generated Swift source as a string literal.
    ///
    /// Multi-line strings need newlines converted to `\n` escape sequences, and special characters
    /// (backslashes and quotes) must be escaped.
    private static func makeSwiftStringLiteralExpression(_ value: String) -> String {
        let escaped =
            value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        return "\"\(escaped)\""
    }

    private static func buildGuidesArray(for property: PropertyInfo) -> String {
        let baseType = property.type.replacingOccurrences(of: "?", with: "")
        var guides: [String] = []

        if baseType.hasPrefix("[") && baseType.hasSuffix("]") && !isDictionaryType(baseType) {
            let minCount = property.guide.constraints.minimumCount
            let maxCount = property.guide.constraints.maximumCount

            if let min = minCount, let max = maxCount {
                if min == max {
                    guides.append(".count(\(min))")
                } else {
                    guides.append(".count(\(min)...\(max))")
                }
            } else {
                if let min = minCount {
                    guides.append(".minimumCount(\(min))")
                }
                if let max = maxCount {
                    guides.append(".maximumCount(\(max))")
                }
            }

            return guides.isEmpty ? "[]" : "[\(guides.joined(separator: ", "))]"
        }

        if baseType == "Int" || baseType == "Double" || baseType == "Float" {
            let minValue = property.guide.constraints.minimum
            let maxValue = property.guide.constraints.maximum

            if let min = minValue, let max = maxValue {
                if let minExpr = numericLiteralExpression(for: baseType, value: min),
                    let maxExpr = numericLiteralExpression(for: baseType, value: max)
                {
                    guides.append(".range(\(minExpr)...\(maxExpr))")
                }
            } else {
                if let min = minValue, let minExpr = numericLiteralExpression(for: baseType, value: min) {
                    guides.append(".minimum(\(minExpr))")
                }
                if let max = maxValue, let maxExpr = numericLiteralExpression(for: baseType, value: max) {
                    guides.append(".maximum(\(maxExpr))")
                }
            }

            return guides.isEmpty ? "[]" : "[\(guides.joined(separator: ", "))]"
        }

        if baseType == "String", !property.guide.constraints.stringGuides.isEmpty {
            return "[\(property.guide.constraints.stringGuides.joined(separator: ", "))]"
        }

        return "[]"
    }

    private static func numericLiteralExpression(for baseType: String, value: Double) -> String? {
        switch baseType {
        case "Int":
            guard value.rounded() == value else { return nil }
            return String(Int(value))
        case "Float":
            return String(Float(value))
        default:
            return String(value)
        }
    }

    private static func parseNumericLiteral(_ expression: ExprSyntax) -> Double? {
        if let intLiteral = expression.as(IntegerLiteralExprSyntax.self) {
            return Double(intLiteral.literal.text)
        } else if let floatLiteral = expression.as(FloatLiteralExprSyntax.self) {
            return Double(floatLiteral.literal.text)
        } else if let prefixExpression = expression.as(PrefixOperatorExprSyntax.self),
            prefixExpression.operator.text == "-"
        {
            if let value = parseNumericLiteral(prefixExpression.expression) {
                return -value
            }
        }
        return nil
    }

    private static func parseClosedRangeInt(_ expression: SequenceExprSyntax) -> (Int?, Int?) {
        let elements = Array(expression.elements)
        guard elements.count == 3,
            let lowerBound = elements[0].as(IntegerLiteralExprSyntax.self),
            let upperBound = elements[2].as(IntegerLiteralExprSyntax.self)
        else { return (nil, nil) }
        return (Int(lowerBound.literal.text), Int(upperBound.literal.text))
    }

    private static func parseClosedRangeDouble(_ expression: SequenceExprSyntax) -> (Double?, Double?) {
        let elements = Array(expression.elements)
        guard elements.count == 3 else { return (nil, nil) }
        let minimum = parseNumericLiteral(elements[0])
        let maximum = parseNumericLiteral(elements[2])
        return (minimum, maximum)
    }

    private static func extractDictionaryTypes(_ type: String) -> (key: String, value: String)? {
        let trimmed = type.trimmingCharacters(in: .whitespacesAndNewlines)

        guard trimmed.hasPrefix("[") && trimmed.hasSuffix("]") else {
            return nil
        }

        let inner = String(trimmed.dropFirst().dropLast())
        guard let colonIndex = topLevelColonIndex(in: inner) else {
            return nil
        }

        let key = inner[..<colonIndex].trimmingCharacters(in: .whitespacesAndNewlines)
        let value = inner[inner.index(after: colonIndex)...]
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !key.isEmpty && !value.isEmpty else {
            return nil
        }

        return (key: key, value: value)
    }

    private static func isDictionaryType(_ type: String) -> Bool {
        extractDictionaryTypes(type) != nil
    }

    private static func baseTypeName(_ type: String) -> String {
        let trimmed = type.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasSuffix("?") {
            return String(trimmed.dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return trimmed
    }

    private static func arrayElementType(from type: String) -> String? {
        let trimmed = type.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("[") && trimmed.hasSuffix("]") {
            let inner = String(trimmed.dropFirst().dropLast())
            guard topLevelColonIndex(in: inner) == nil else {
                return nil
            }
            return inner.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if trimmed.hasPrefix("Array<") && trimmed.hasSuffix(">") {
            return String(trimmed.dropFirst("Array<".count).dropLast())
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return nil
    }

    private static let primitiveTypes: Set<String> = [
        "String",
        "Int",
        "Double",
        "Float",
        "Bool",
        "Decimal",
    ]

    private static func partiallyGeneratedTypeName(for type: String) -> String {
        partiallyGeneratedTypeName(for: type, preserveOptional: false)
    }

    private static func partiallyGeneratedTypeName(for type: String, preserveOptional: Bool) -> String {
        let trimmed = type.trimmingCharacters(in: .whitespacesAndNewlines)
        if preserveOptional {
            var normalized = trimmed
            var optionalCount = 0
            while normalized.hasSuffix("?") {
                normalized = String(normalized.dropLast())
                optionalCount += 1
            }
            if optionalCount > 1 {
                return "\(partiallyGeneratedTypeName(for: normalized, preserveOptional: false))?"
            }
            if optionalCount == 1 {
                return "\(partiallyGeneratedTypeName(for: normalized, preserveOptional: true))?"
            }
        }

        let baseType = baseTypeName(trimmed)
        if primitiveTypes.contains(baseType) || isDictionaryType(baseType) {
            return baseType
        }
        if let elementType = arrayElementType(from: baseType) {
            let elementPartial = partiallyGeneratedTypeName(for: elementType, preserveOptional: true)
            return "[\(elementPartial)]"
        }
        return "\(baseType).PartiallyGenerated"
    }

    private static func getDefaultValue(for type: String) -> String {
        let trimmedType = type.trimmingCharacters(in: .whitespacesAndNewlines)

        if trimmedType.hasSuffix("?") {
            return "nil"
        }

        if isDictionaryType(trimmedType) {
            return "[:]"
        }

        if trimmedType.hasPrefix("[") && trimmedType.hasSuffix("]") {
            return "[]"
        }

        switch trimmedType {
        case "String":
            return "\"\""
        case "Int":
            return "0"
        case "Double", "Float":
            return "0.0"
        case "Bool":
            return "false"
        default:
            return "nil"
        }
    }

    private static func generatePropertyAssignment(for property: PropertyInfo) -> String {
        let propertyName = property.name
        let propertyType = property.type.trimmingCharacters(in: .whitespacesAndNewlines)
        let defaultValue = getDefaultValue(for: propertyType)

        switch propertyType {
        case "String":
            return
                "self.\(propertyName) = (json[\"\(propertyName)\"] as? String) ?? \(defaultValue)"
        case "Int":
            return "self.\(propertyName) = (json[\"\(propertyName)\"] as? Int) ?? \(defaultValue)"
        case "Double":
            return
                "self.\(propertyName) = (json[\"\(propertyName)\"] as? Double) ?? \(defaultValue)"
        case "Float":
            return
                "self.\(propertyName) = Float((json[\"\(propertyName)\"] as? Double) ?? Double(\(defaultValue)))"
        case "Bool":
            return "self.\(propertyName) = (json[\"\(propertyName)\"] as? Bool) ?? \(defaultValue)"
        default:
            return "self.\(propertyName) = \(defaultValue)"
        }
    }

    private static func generateRawContentProperty() -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                private let _rawGeneratedContent: GeneratedContent
                """
        )
    }

    private static func generateMemberwiseInit(properties: [PropertyInfo], explicitNil: Bool) -> DeclSyntax {
        if properties.isEmpty {
            return DeclSyntax(
                stringLiteral: """
                    nonisolated public init() {
                        self._rawGeneratedContent = GeneratedContent(kind: .structure(properties: [:], orderedKeys: []))
                    }
                    """
            )
        }

        let parameters = properties.map { prop in
            "\(prop.name): \(prop.type)"
        }.joined(separator: ", ")

        let assignments = properties.map { prop in
            "self.\(prop.name) = \(prop.name)"
        }.joined(separator: "\n        ")

        let propertyConversions = properties.map { prop in
            let propName = prop.name
            let propType = prop.type

            if propType.hasSuffix("?") {
                return optionalPropertyConversion(
                    propertyName: propName,
                    baseType: String(propType.dropLast()),
                    explicitNil: explicitNil
                )
            } else if isDictionaryType(propType) {
                return "properties[\"\(propName)\"] = \(propName).generatedContent"
            } else if propType.hasPrefix("[") && propType.hasSuffix("]") {
                return "properties[\"\(propName)\"] = GeneratedContent(elements: \(propName))"
            } else {
                switch propType {
                case "String":
                    return "properties[\"\(propName)\"] = GeneratedContent(\(propName))"
                case "Int", "Double", "Float", "Bool", "Decimal":
                    return "properties[\"\(propName)\"] = \(propName).generatedContent"
                default:
                    return "properties[\"\(propName)\"] = \(propName).generatedContent"
                }
            }
        }.joined(separator: "\n        ")

        let orderedKeys = properties.map { "\"\($0.name)\"" }.joined(separator: ", ")

        return DeclSyntax(
            stringLiteral: """
                nonisolated public init(\(parameters)) {
                    \(assignments)
                    
                    var properties: [String: GeneratedContent] = [:]
                    \(propertyConversions)
                    
                    self._rawGeneratedContent = GeneratedContent(
                        kind: .structure(
                            properties: properties,
                            orderedKeys: [\(orderedKeys)].filter { properties[$0] != nil }
                        )
                    )
                }
                """
        )
    }

    private static func generateInitFromGeneratedContent(
        structName: String,
        properties: [PropertyInfo]
    ) -> DeclSyntax {
        let propertyExtractions = properties.map { prop in
            generatePropertyExtraction(propertyName: prop.name, propertyType: prop.type)
        }.joined(separator: "\n            ")

        if properties.isEmpty {
            return DeclSyntax(
                stringLiteral: """
                    nonisolated public init(_ generatedContent: GeneratedContent) throws {
                        self._rawGeneratedContent = generatedContent

                        guard case .structure = generatedContent.kind else {
                            throw DecodingError.typeMismatch(
                                \(structName).self,
                                DecodingError.Context(codingPath: [], debugDescription: "Expected structure for \(structName)")
                            )
                        }
                    }
                    """
            )
        } else {
            return DeclSyntax(
                stringLiteral: """
                    nonisolated public init(_ generatedContent: GeneratedContent) throws {
                        self._rawGeneratedContent = generatedContent

                        guard case .structure(let properties, _) = generatedContent.kind else {
                            throw DecodingError.typeMismatch(
                                \(structName).self,
                                DecodingError.Context(codingPath: [], debugDescription: "Expected structure for \(structName)")
                            )
                        }

                        \(propertyExtractions)
                    }
                    """
            )
        }
    }

    private static func generatePartialPropertyExtraction(
        propertyName: String,
        propertyType: String
    ) -> String {
        let baseType = baseTypeName(propertyType)

        switch baseType {
        case "String":
            return "self.\(propertyName) = try? properties[\"\(propertyName)\"]?.value(String.self)"
        case "Int":
            return "self.\(propertyName) = try? properties[\"\(propertyName)\"]?.value(Int.self)"
        case "Double":
            return "self.\(propertyName) = try? properties[\"\(propertyName)\"]?.value(Double.self)"
        case "Float":
            return "self.\(propertyName) = try? properties[\"\(propertyName)\"]?.value(Float.self)"
        case "Bool":
            return "self.\(propertyName) = try? properties[\"\(propertyName)\"]?.value(Bool.self)"
        default:
            if isDictionaryType(baseType) {
                return """
                    if let value = properties[\"\(propertyName)\"] {
                        self.\(propertyName) = try? \(baseType)(value)
                    } else {
                        self.\(propertyName) = nil
                    }
                    """
            } else if let elementType = arrayElementType(from: baseType) {
                let elementPartial = partiallyGeneratedTypeName(for: elementType, preserveOptional: true)
                let arrayPartial = "[\(elementPartial)]"
                return """
                    if let value = properties[\"\(propertyName)\"] {
                        self.\(propertyName) = try? \(arrayPartial)(value)
                    } else {
                        self.\(propertyName) = nil
                    }
                    """
            } else {
                let partialType = partiallyGeneratedTypeName(for: baseType)
                return """
                    if let value = properties[\"\(propertyName)\"] {
                        self.\(propertyName) = try? \(partialType)(value)
                    } else {
                        self.\(propertyName) = nil
                    }
                    """
            }
        }
    }

    private static func generatePropertyExtraction(propertyName: String, propertyType: String)
        -> String
    {
        switch propertyType {
        case "String":
            return """
                self.\(propertyName) = try properties["\(propertyName)"]?.value(String.self) ?? ""
                """
        case "Int":
            return """
                self.\(propertyName) = try properties["\(propertyName)"]?.value(Int.self) ?? 0
                """
        case "Double":
            return """
                self.\(propertyName) = try properties["\(propertyName)"]?.value(Double.self) ?? 0.0
                """
        case "Float":
            return """
                self.\(propertyName) = try properties["\(propertyName)"]?.value(Float.self) ?? 0.0
                """
        case "Bool":
            return """
                self.\(propertyName) = try properties["\(propertyName)"]?.value(Bool.self) ?? false
                """
        default:
            let isOptional = propertyType.hasSuffix("?")
            let isDictionary = isDictionaryType(
                propertyType.replacingOccurrences(of: "?", with: "")
            )
            let isArray =
                !isDictionary && propertyType.hasPrefix("[") && propertyType.hasSuffix("]")

            if isOptional {
                let baseType = propertyType.replacingOccurrences(of: "?", with: "")

                if baseType == "Int" || baseType == "String" || baseType == "Double"
                    || baseType == "Float" || baseType == "Bool"
                {
                    return """
                        if let value = properties["\(propertyName)"] {
                            switch value.kind {
                            case .null:
                                self.\(propertyName) = nil
                            default:
                                self.\(propertyName) = try value.value(\(baseType).self)
                            }
                        } else {
                            self.\(propertyName) = nil
                        }
                        """
                } else {
                    return """
                        if let value = properties["\(propertyName)"] {
                            switch value.kind {
                            case .null:
                                self.\(propertyName) = nil
                            default:
                                self.\(propertyName) = try \(baseType)(value)
                            }
                        } else {
                            self.\(propertyName) = nil
                        }
                        """
                }

            } else if isDictionary {
                return """
                    if let value = properties["\(propertyName)"] {
                        self.\(propertyName) = try \(propertyType)(value)
                    } else {
                        self.\(propertyName) = [:]
                    }
                    """
            } else if isArray {
                return """
                    if let value = properties["\(propertyName)"] {
                        self.\(propertyName) = try \(propertyType)(value)
                    } else {
                        self.\(propertyName) = []
                    }
                    """
            } else {
                return """
                    if let value = properties["\(propertyName)"] {
                        self.\(propertyName) = try \(propertyType)(value)
                    } else {
                        self.\(propertyName) = try \(propertyType)(GeneratedContent("{}"))
                    }
                    """
            }
        }
    }

    private static func generateGeneratedContentProperty(
        structName: String,
        description: String?,
        properties: [PropertyInfo],
        explicitNil: Bool
    ) -> DeclSyntax {
        let propertyConversions = properties.map { prop in
            let propName = prop.name
            let propType = prop.type

            if propType.hasSuffix("?") {
                return optionalPropertyConversion(
                    propertyName: propName,
                    baseType: String(propType.dropLast()),
                    explicitNil: explicitNil
                )
            } else if isDictionaryType(propType) {
                return "properties[\"\(propName)\"] = \(propName).generatedContent"
            } else if propType.hasPrefix("[") && propType.hasSuffix("]") {
                return "properties[\"\(propName)\"] = GeneratedContent(elements: \(propName))"
            } else {
                switch propType {
                case "String":
                    return "properties[\"\(propName)\"] = GeneratedContent(\(propName))"
                case "Int", "Double", "Float", "Bool", "Decimal":
                    return "properties[\"\(propName)\"] = \(propName).generatedContent"
                default:
                    return "properties[\"\(propName)\"] = \(propName).generatedContent"
                }
            }
        }.joined(separator: "\n            ")

        let orderedKeys = properties.map { "\"\($0.name)\"" }.joined(separator: ", ")

        if properties.isEmpty {
            return DeclSyntax(
                stringLiteral: """
                    nonisolated public var generatedContent: GeneratedContent {
                        let properties: [String: GeneratedContent] = [:]

                        return GeneratedContent(
                            kind: .structure(
                                properties: properties,
                                orderedKeys: []
                            )
                        )
                    }
                    """
            )
        } else {
            return DeclSyntax(
                stringLiteral: """
                    nonisolated public var generatedContent: GeneratedContent {
                        var properties: [String: GeneratedContent] = [:]
                        \(propertyConversions)

                        return GeneratedContent(
                            kind: .structure(
                                properties: properties,
                                orderedKeys: [\(orderedKeys)].filter { properties[$0] != nil }
                            )
                        )
                    }
                    """
            )
        }
    }

    private static func generateGenerationSchemaProperty(
        structName: String,
        description: String?,
        properties: [PropertyInfo],
        explicitNil: Bool
    ) -> DeclSyntax {
        let propertySchemas = properties.map { prop in
            let escapedDescription = escapeDescriptionString(prop.guide.description)
            let guidesArray = buildGuidesArray(for: prop)

            return """
                GenerationSchema.Property(
                                name: "\(prop.name)",
                                description: \(escapedDescription),
                                type: \(prop.type).self,
                                guides: \(guidesArray)
                            )
                """
        }.joined(separator: ",\n            ")

        return DeclSyntax(
            stringLiteral: """
                nonisolated public static var generationSchema: GenerationSchema {
                    return GenerationSchema(
                        type: Self.self,
                        description: \(description.map { "\"\($0)\"" } ?? "\"Generated \(structName)\""),\(explicitNil ? "\n            representNilExplicitlyInGeneratedContent: true," : "")
                        properties: [\(properties.isEmpty ? "" : "\n            \(propertySchemas)\n        ")]
                    )
                }
                """
        )
    }

    private static func generateAsPartiallyGeneratedMethod(structName: String) -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                nonisolated public func asPartiallyGenerated() -> PartiallyGenerated {
                    return try! PartiallyGenerated(_rawGeneratedContent)
                }
                """
        )
    }

    private static func generateAsPartiallyGeneratedMethodForEnum(enumName: String) -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                nonisolated public func asPartiallyGenerated() -> \(enumName) {
                    return self
                }
                """
        )
    }

    private static func generatePartiallyGeneratedStruct(
        structName: String,
        properties: [PropertyInfo]
    ) -> DeclSyntax {
        let optionalProperties = properties.map { prop in
            let partialType = partiallyGeneratedTypeName(for: prop.type)
            return "public let \(prop.name): \(partialType)?"
        }.joined(separator: "\n        ")

        let propertyExtractions = properties.map { prop in
            generatePartialPropertyExtraction(propertyName: prop.name, propertyType: prop.type)
        }.joined(separator: "\n            ")

        return DeclSyntax(
            stringLiteral: """
                public struct PartiallyGenerated: Identifiable, Sendable, ConvertibleFromGeneratedContent {
                    public var id: GenerationID

                    \(optionalProperties)

                    private let rawContent: GeneratedContent

                    public init(_ generatedContent: GeneratedContent) throws {
                        self.id = generatedContent.id ?? GenerationID()
                        self.rawContent = generatedContent

                        if \(properties.isEmpty ? "case .structure = generatedContent.kind" : "case .structure(let properties, _) = generatedContent.kind") {
                            \(propertyExtractions)
                        } else {
                            \(properties.map { "self.\($0.name) = nil" }.joined(separator: "\n                    "))
                        }
                    }

                    public var generatedContent: GeneratedContent {
                        return rawContent
                    }
                }
                """
        )
    }

    private static func generateInstructionsRepresentationProperty() -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                nonisolated public var instructionsRepresentation: Instructions {
                    return Instructions(self.generatedContent.jsonString)
                }
                """
        )
    }

    private static func generatePromptRepresentationProperty() -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                nonisolated public var promptRepresentation: Prompt {
                    return Prompt(self.generatedContent.jsonString)
                }
                """
        )
    }

    private static func extractEnumCases(from enumDecl: EnumDeclSyntax) -> [EnumCaseInfo] {
        var cases: [EnumCaseInfo] = []

        for member in enumDecl.memberBlock.members {
            if let caseDecl = member.decl.as(EnumCaseDeclSyntax.self) {
                for element in caseDecl.elements {
                    let caseName = element.name.text
                    var associatedValues: [(label: String?, type: String)] = []

                    if let parameterClause = element.parameterClause {
                        for parameter in parameterClause.parameters {
                            let label = parameter.firstName?.text
                            let type = parameter.type.description.trimmingCharacters(
                                in: .whitespacesAndNewlines
                            )
                            associatedValues.append((label: label, type: type))
                        }
                    }

                    let guideDescription: String? = nil

                    cases.append(
                        EnumCaseInfo(
                            name: caseName,
                            associatedValues: associatedValues,
                            guideDescription: guideDescription
                        )
                    )
                }
            }
        }

        return cases
    }

    private static func generateEnumInitFromGeneratedContent(
        enumName: String,
        cases: [EnumCaseInfo]
    ) -> DeclSyntax {
        let hasAnyAssociatedValues = cases.contains { $0.hasAssociatedValues }

        if hasAnyAssociatedValues {
            let switchCases = cases.map { enumCase in
                if enumCase.associatedValues.isEmpty {
                    return """
                        case "\(enumCase.name)":
                            self = .\(enumCase.name)
                        """
                } else if enumCase.isSingleUnlabeledValue {
                    let valueType = enumCase.associatedValues[0].type
                    return generateSingleValueCase(caseName: enumCase.name, valueType: valueType)
                } else {
                    return generateMultipleValueCase(
                        caseName: enumCase.name,
                        associatedValues: enumCase.associatedValues
                    )
                }
            }.joined(separator: "\n                ")

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public init(_ generatedContent: GeneratedContent) throws {
                        do {
                            guard case .structure(let properties, _) = generatedContent.kind else {
                                throw DecodingError.typeMismatch(
                                    \(enumName).self,
                                    DecodingError.Context(codingPath: [], debugDescription: "Expected structure for enum \(enumName)")
                                )
                            }

                            guard case .string(let caseValue) = properties["case"]?.kind else {
                                struct Key: CodingKey {
                                    var stringValue: String
                                    var intValue: Int? { nil }
                                    init(stringValue: String) { self.stringValue = stringValue }
                                    init?(intValue: Int) { nil }
                                }
                                throw DecodingError.keyNotFound(
                                    Key(stringValue: "case"),
                                    DecodingError.Context(codingPath: [], debugDescription: "Missing 'case' property in enum data for \(enumName)")
                                )
                            }

                            let valueContent = properties["value"]

                            switch caseValue {
                            \(switchCases)
                            default:
                                throw DecodingError.dataCorrupted(
                                    DecodingError.Context(codingPath: [], debugDescription: "Invalid enum case '\\(caseValue)' for \(enumName). Valid cases: [\(cases.map { $0.name }.joined(separator: ", "))]")
                                )
                            }
                        } catch {
                            guard case .string(let value) = generatedContent.kind else {
                                throw error
                            }
                            let trimmedValue = value.trimmingCharacters(in: .whitespacesAndNewlines)
                            switch trimmedValue {
                            \(cases.filter { !$0.hasAssociatedValues }.map { "case \"\($0.name)\": self = .\($0.name)" }.joined(separator: "\n                    "))
                            default:
                                throw DecodingError.dataCorrupted(
                                    DecodingError.Context(codingPath: [], debugDescription: "Invalid enum case '\\(trimmedValue)' for \(enumName). Valid cases: [\(cases.map { $0.name }.joined(separator: ", "))]")
                                )
                            }
                        }
                    }
                    """
            )
        } else {
            let switchCases = cases.map { enumCase in
                "case \"\(enumCase.name)\": self = .\(enumCase.name)"
            }.joined(separator: "\n            ")

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public init(_ generatedContent: GeneratedContent) throws {
                        guard case .string(let value) = generatedContent.kind else {
                            throw DecodingError.typeMismatch(
                                \(enumName).self,
                                DecodingError.Context(codingPath: [], debugDescription: "Expected string for enum \(enumName)")
                            )
                        }
                        let trimmedValue = value.trimmingCharacters(in: .whitespacesAndNewlines)

                        switch trimmedValue {
                        \(switchCases)
                        default:
                            throw DecodingError.dataCorrupted(
                                DecodingError.Context(codingPath: [], debugDescription: "Invalid enum case '\\(trimmedValue)' for \(enumName). Valid cases: [\(cases.map { $0.name }.joined(separator: ", "))]")
                            )
                        }
                    }
                    """
            )
        }
    }

    private static func generateSingleValueCase(caseName: String, valueType: String) -> String {
        switch valueType {
        case "String":
            return """
                case "\(caseName)":
                    if let valueContent = valueContent,
                       case .string(let stringValue) = valueContent.kind {
                        self = .\(caseName)(stringValue)
                    } else {
                        self = .\(caseName)("")
                    }
                """
        case "Int":
            return """
                case "\(caseName)":
                    if let valueContent = valueContent {
                        let intValue = try valueContent.value(Int.self)
                        self = .\(caseName)(intValue)
                    } else {
                        self = .\(caseName)(0)
                    }
                """
        case "Double":
            return """
                case "\(caseName)":
                    if let valueContent = valueContent {
                        let doubleValue = try valueContent.value(Double.self)
                        self = .\(caseName)(doubleValue)
                    } else {
                        self = .\(caseName)(0.0)
                    }
                """
        case "Bool":
            return """
                case "\(caseName)":
                    if let valueContent = valueContent {
                        let boolValue = try valueContent.value(Bool.self)
                        self = .\(caseName)(boolValue)
                    } else {
                        self = .\(caseName)(false)
                    }
                """
        default:
            return """
                case "\(caseName)":
                    if let valueContent = valueContent {
                        let associatedValue = try \(valueType)(valueContent)
                        self = .\(caseName)(associatedValue)
                    } else {
                        throw DecodingError.valueNotFound(
                            \(valueType).self,
                            DecodingError.Context(codingPath: [], debugDescription: "Missing value for enum case '\(caseName)' with associated type \(valueType)")
                        )
                    }
                """
        }
    }

    private static func generateMultipleValueCase(
        caseName: String,
        associatedValues: [(label: String?, type: String)]
    ) -> String {
        let valueExtractions = associatedValues.enumerated().map { index, assocValue in
            let label = assocValue.label ?? "param\(index)"
            let type = assocValue.type

            switch type {
            case "String":
                return "let \(label) = try valueProperties[\"\(label)\"]?.value(String.self) ?? \"\""
            case "Int":
                return "let \(label) = try valueProperties[\"\(label)\"]?.value(Int.self) ?? 0"
            case "Double":
                return "let \(label) = try valueProperties[\"\(label)\"]?.value(Double.self) ?? 0.0"
            case "Bool":
                return "let \(label) = try valueProperties[\"\(label)\"]?.value(Bool.self) ?? false"
            default:
                return
                    "let \(label) = try \(type)(valueProperties[\"\(label)\"] ?? GeneratedContent(\"{}\"))"
            }
        }.joined(separator: "\n                    ")

        let parameterList = associatedValues.enumerated().map { index, assocValue in
            let label = assocValue.label ?? "param\(index)"
            if assocValue.label != nil {
                return "\(label): \(label)"
            } else {
                return label
            }
        }.joined(separator: ", ")

        return """
            case "\(caseName)":
                if let valueContent = valueContent {
                    guard case .structure(let valueProperties, _) = valueContent.kind else {
                        throw DecodingError.typeMismatch(
                            [String: Any].self,
                            DecodingError.Context(codingPath: [], debugDescription: "Expected structure for enum case '\(caseName)' associated values")
                        )
                    }
                    \(valueExtractions)
                    self = .\(caseName)(\(parameterList))
                } else {
                    throw DecodingError.valueNotFound(
                        [String: Any].self,
                        DecodingError.Context(codingPath: [], debugDescription: "Missing value data for enum case '\(caseName)' with associated values")
                    )
                }
            """
    }

    private static func generateEnumGeneratedContentProperty(
        enumName: String,
        description: String?,
        cases: [EnumCaseInfo]
    ) -> DeclSyntax {
        let hasAnyAssociatedValues = cases.contains { $0.hasAssociatedValues }

        if hasAnyAssociatedValues {
            let switchCases = cases.map { enumCase in
                if enumCase.associatedValues.isEmpty {
                    return """
                        case .\(enumCase.name):
                            return GeneratedContent(properties: [
                                "case": GeneratedContent("\(enumCase.name)"),
                                "value": GeneratedContent("")
                            ])
                        """
                } else if enumCase.isSingleUnlabeledValue {
                    return """
                        case .\\(enumCase.name)(let value):
                            return GeneratedContent(properties: [
                                "case": GeneratedContent("\\(enumCase.name)"),
                                "value": GeneratedContent("\\\\(value)")
                            ])
                        """
                } else {
                    return generateMultipleValueSerialization(
                        caseName: enumCase.name,
                        associatedValues: enumCase.associatedValues
                    )
                }
            }.joined(separator: "\n            ")

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public var generatedContent: GeneratedContent {
                        switch self {
                        \(switchCases)
                        }
                    }
                    """
            )
        } else {
            let switchCases = cases.map { enumCase in
                "case .\(enumCase.name): return GeneratedContent(\"\(enumCase.name)\")"
            }.joined(separator: "\n            ")

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public var generatedContent: GeneratedContent {
                        switch self {
                        \(switchCases)
                        }
                    }
                    """
            )
        }
    }

    private static func generateSingleValueSerialization(caseName: String, valueType: String)
        -> String
    {
        switch valueType {
        case "String", "Int", "Double", "Bool":
            return """
                case .\(caseName)(let value):
                    return GeneratedContent(properties: [
                        "case": GeneratedContent("\(caseName)"),
                        "value": GeneratedContent("\\(value)")
                    ])
                """
        default:
            return """
                case .\(caseName)(let value):
                    return GeneratedContent(properties: [
                        "case": GeneratedContent("\(caseName)"),
                        "value": value.generatedContent
                    ])
                """
        }
    }

    private static func generateMultipleValueSerialization(
        caseName: String,
        associatedValues: [(label: String?, type: String)]
    ) -> String {
        let parameterList = associatedValues.enumerated().map { index, assocValue in
            let label = assocValue.label ?? "param\(index)"
            return "let \(label)"
        }.joined(separator: ", ")

        let propertyMappings = associatedValues.enumerated().map { index, assocValue in
            let label = assocValue.label ?? "param\(index)"
            let type = assocValue.type

            switch type {
            case "String", "Int", "Double", "Bool":
                return "\"\(label)\": GeneratedContent(\"\\(\(label))\")"
            default:
                return "\"\(label)\": \(label).generatedContent"
            }
        }.joined(separator: ",\n                        ")

        return """
            case .\(caseName)(\(parameterList)):
                return GeneratedContent(properties: [
                    "case": GeneratedContent("\(caseName)"),
                    "value": GeneratedContent(properties: [
                        \(propertyMappings)
                    ])
                ])
            """
    }

    private static func generateEnumFromGeneratedContentMethod(enumName: String) -> DeclSyntax {
        return DeclSyntax(
            stringLiteral: """
                public static func from(generatedContent: GeneratedContent) throws -> \(enumName) {
                    return try \(enumName)(generatedContent)
                }
                """
        )
    }

    private static func generateEnumGenerationSchemaProperty(
        enumName: String,
        description: String?,
        cases: [EnumCaseInfo]
    ) -> DeclSyntax {
        let hasAnyAssociatedValues = cases.contains { $0.hasAssociatedValues }

        if hasAnyAssociatedValues {

            let caseProperty = """
                GenerationSchema.Property(
                                        name: "case",
                                        description: "Enum case identifier",
                                        type: String.self,
                                        guides: []
                                    )
                """
            let valueProperty = """
                GenerationSchema.Property(
                                        name: "value",
                                        description: "Associated value data",
                                        type: String.self,
                                        guides: []
                                    )
                """

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public static var generationSchema: GenerationSchema {
                        return GenerationSchema(
                            type: Self.self,
                            description: \(description.map { "\"\($0)\"" } ?? "\"Generated \(enumName)\""),
                            properties: [
                                \(caseProperty),
                                \(valueProperty)
                            ]
                        )
                    }
                    """
            )
        } else {
            let caseNames = cases.map { "\"\($0.name)\"" }.joined(separator: ", ")

            return DeclSyntax(
                stringLiteral: """
                    nonisolated public static var generationSchema: GenerationSchema {
                        return GenerationSchema(
                            type: Self.self,
                            description: \(description.map { "\"\($0)\"" } ?? "\"Generated \(enumName)\""),
                            anyOf: [\(caseNames)]
                        )
                    }
                    """
            )
        }
    }
}

// MARK: - Error

public enum GenerableMacroError: Error, CustomStringConvertible {
    case notApplicableToType
    case invalidSyntax
    case missingRequiredParameter
    case nonLiteralRepresentNilExplicitly

    public var description: String {
        switch self {
        case .notApplicableToType:
            return "@Generable can only be applied to structs, actors, or enumerations"
        case .invalidSyntax:
            return "Invalid macro syntax"
        case .missingRequiredParameter:
            return "Missing required parameter"
        case .nonLiteralRepresentNilExplicitly:
            return "representNilExplicitlyInGeneratedContent must be a Boolean literal (true or false)"
        }
    }
}

// MARK: -

private struct EnumCaseInfo {
    let name: String
    let associatedValues: [(label: String?, type: String)]
    let guideDescription: String?

    var hasAssociatedValues: Bool {
        !associatedValues.isEmpty
    }

    var isSingleUnlabeledValue: Bool {
        associatedValues.count == 1 && associatedValues[0].label == nil
    }

    var isMultipleLabeledValues: Bool {
        associatedValues.count > 1
            || (associatedValues.count == 1 && associatedValues[0].label != nil)
    }
}

private struct GuideInfo {
    let description: String?
    let constraints: Constraints
}

private struct Constraints {
    var minimumCount: Int?
    var maximumCount: Int?
    var minimum: Double?
    var maximum: Double?
    var pattern: String?
    var stringGuides: [String] = []
}

private struct PropertyInfo {
    let name: String
    let type: String
    let guide: GuideInfo
}
