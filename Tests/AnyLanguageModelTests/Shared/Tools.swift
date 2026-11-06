import AnyLanguageModel

struct WeatherTool: Tool {
    let name = "getWeather"
    let description = "Retrieve the latest weather information for a city"

    @Generable
    struct Arguments {
        @Guide(description: "The city to fetch the weather for")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        "The weather in \(arguments.city) is sunny and 72°F / 23°C"
    }
}

struct ClockTool: Tool {
    let name = "getTime"
    let description = "Retrieve the time in a city"

    @Generable
    struct Arguments {
        @Guide(description: "The city to fetch the time for")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        "It's noon in \(arguments.city)"
    }
}

/// A structured answer that a model can only fill in correctly
/// with the output of ``WeatherTool``.
@Generable
struct WeatherReport {
    @Guide(description: "The city that the weather is for")
    var city: String

    @Guide(description: "The temperature in degrees Fahrenheit")
    var temperatureFahrenheit: Int
}

/// A delegate that stops every tool call before it runs.
struct StopToolExecutionDelegate: ToolExecutionDelegate {
    func toolCallDecision(for toolCall: Transcript.ToolCall, in session: LanguageModelSession) async
        -> ToolExecutionDecision
    {
        .stop
    }
}
