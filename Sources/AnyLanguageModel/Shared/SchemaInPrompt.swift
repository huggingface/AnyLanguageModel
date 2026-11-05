import Foundation
import JSONSchema

extension GenerationSchema {
    /// What a request says after its prompt of the schema its response must match,
    /// when `includeSchemaInPrompt` asks for it: `nil` for free text, which has no schema.
    ///
    /// A model whose server enforces a schema still benefits from reading it,
    /// and one whose server ignores it, as some OpenAI-compatible APIs and Ollama's cloud models
    /// do, only knows what to write from this.
    static func promptInstructions<Content: Generable>(
        generating type: Content.Type,
        schema: GenerationSchema,
        includeSchemaInPrompt: Bool
    ) -> String? {
        guard includeSchemaInPrompt, type != String.self else { return nil }
        return schema.schemaPrompt()
    }

    /// The schema in words, with its JSON Schema: inlined, as a model reads it best, or, when it
    /// can't be, as a recursive schema can't, as it's encoded, with its definitions.
    func schemaPrompt() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let inlined = try? inlinedJSONSchema(constantsAsEnums: false)
        let data: Data? =
            if let inlined { try? encoder.encode(inlined) } else { try? encoder.encode(self) }
        guard let data, let schemaJSON = String(data: data, encoding: .utf8) else {
            return "Respond with valid JSON only."
        }
        guard let jsonSchema = inlined ?? (try? JSONDecoder().decode(JSONSchema.self, from: data)) else {
            return "Respond with valid JSON matching this schema:\n\(schemaJSON)"
        }

        var header = "Respond with valid JSON matching this \(jsonSchema.typeName) schema"
        if let description = jsonSchema.description, !description.isEmpty {
            header += " (\(description))"
        }
        if let constValue = jsonSchema.const,
            let data = try? encoder.encode(constValue),
            let constString = String(data: data, encoding: .utf8)
        {
            header += ". Expected value: \(constString)"
        } else if let enumValues = jsonSchema.enum, !enumValues.isEmpty,
            let data = try? encoder.encode(enumValues),
            let enumString = String(data: data, encoding: .utf8)
        {
            header += ". Allowed values: \(enumString)"
        }
        return "\(header):\n\(schemaJSON)"
    }
}

extension Transcript {
    /// The transcript with `instructions` said after the prompt being answered: the last one.
    ///
    /// Said there, next to what's asked, a model follows them even after rounds of tool calls,
    /// where it overlooks a system message; and what comes before the prompt stays as it was.
    func saying(_ instructions: String?) -> Transcript {
        guard let instructions else { return self }
        var entries = Array(self)
        guard let index = entries.lastIndex(where: { if case .prompt = $0 { true } else { false } }),
            case .prompt(var prompt) = entries[index]
        else { return self }
        prompt.segments.append(.text(Transcript.TextSegment(content: instructions)))
        entries[index] = .prompt(prompt)
        return Transcript(entries: entries)
    }
}
