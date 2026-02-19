# AnyLanguageModel

A Swift package that provides a drop-in replacement for
[Apple's Foundation Models framework](https://developer.apple.com/documentation/FoundationModels)
with support for custom language model providers.
All you need to do is change your import statement:

```diff
- import FoundationModels
+ import AnyLanguageModel
```

```swift
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

let model = SystemLanguageModel.default
let session = LanguageModelSession(model: model, tools: [WeatherTool()])

let response = try await session.respond {
    Prompt("How's the weather in Cupertino?")
}
print(response.content)
```

To observe or control tool execution, assign a delegate on the session:

```swift
actor ToolExecutionObserver: ToolExecutionDelegate {
    func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: LanguageModelSession) async {
        print("Generated tool calls: \(toolCalls)")
    }

    func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: LanguageModelSession
    ) async -> ToolExecutionDecision {
        // Return .stop to halt after tool calls, or .provideOutput(...) to bypass execution.
        // This is a good place to ask the user for confirmation (for example, in a modal dialog).
        .execute
    }

    func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: LanguageModelSession
    ) async {
        print("Executed tool call: \(toolCall)")
    }
}

let session = LanguageModelSession(model: model, tools: [WeatherTool()])
session.toolExecutionDelegate = ToolExecutionObserver()
```

## Features

### Supported Providers

- [x] [Apple Foundation Models](https://developer.apple.com/documentation/FoundationModels)
- [x] [Core ML](https://developer.apple.com/documentation/coreml) models
- [x] [MLX](https://github.com/ml-explore/mlx-swift) models
- [x] [llama.cpp](https://github.com/ggml-org/llama.cpp) (GGUF models)
- [x] Ollama [HTTP API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [x] Anthropic [Messages API](https://docs.claude.com/en/api/messages)
- [x] Google [Gemini API](https://ai.google.dev/api/generate-content)
- [x] OpenAI [Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [x] OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses)
- [x] [Open Responses](https://www.openresponses.org) (multi-provider Responses API–compatible endpoints)

## Requirements

- Swift 6.1+
- iOS 17.0+ / macOS 14.0+ / visionOS 1.0+ / Linux

> [!IMPORTANT]
> A bug in Xcode 26 may cause build errors
> when targeting macOS 15 / iOS 18 or earlier
> (e.g. `Conformance of 'String' to 'Generable' is only available in macOS 26.0 or newer`).
> As a workaround, build your project with Xcode 16.
> For more information, see [issue #15](https://github.com/mattt/AnyLanguageModel/issues/15).

## Installation

Add this package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/AnyLanguageModel", from: "0.7.0")
]
```

### Package Traits

AnyLanguageModel uses [Swift 6.1 traits](https://docs.swift.org/swiftpm/documentation/packagemanagerdocs/packagetraits/)
to conditionally include heavy dependencies,
allowing you to opt-in only to the language model backends you need.
This results in smaller binary sizes and faster build times.

**Available traits**:

- `CoreML`: Enables Core ML model support
  (depends on `huggingface/swift-transformers`)
- `MLX`: Enables MLX model support
  (depends on `ml-explore/mlx-swift-lm`)
- `Llama`: Enables llama.cpp support
  (requires `mattt/llama.swift`)

By default, no traits are enabled.
To enable specific traits, specify them in your package's dependencies:

```swift
// In your Package.swift
dependencies: [
    .package(
        url: "https://github.com/mattt/AnyLanguageModel.git",
        from: "0.7.0",
        traits: ["CoreML", "MLX"] // Enable CoreML and MLX support
    )
]
```

> [!IMPORTANT]
> Due to a [Swift Package Manager bug](https://github.com/swiftlang/swift-package-manager/issues/9286),
> dependency resolution may fail when you enable traits,
> producing the error "exhausted attempts to resolve the dependencies graph."
> To work around this issue,
> add the underlying dependencies for each trait directly to your package:
>
> ```swift
> dependencies: [
>     .package(
>         url: "https://github.com/mattt/AnyLanguageModel.git",
>         from: "0.7.0",
>         traits: ["CoreML", "MLX", "Llama"]
>     ),
>     .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"), // CoreML
>     .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.25.5"),       // MLX
>     .package(url: "https://github.com/mattt/llama.swift", from: "2.0.0"),              // Llama
> ]
> ```
>
> Include only the dependencies that correspond to the traits you enable.
> For more information, see [issue #135](https://github.com/mattt/AnyLanguageModel/issues/135).

### Using Traits in Xcode Projects

Xcode doesn't yet provide a built-in way to declare package dependencies with traits.
As a workaround,
you can create an internal Swift package that acts as a shim,
exporting the `AnyLanguageModel` module with the desired traits enabled.
Your Xcode project can then add this internal package as a local dependency.

For example,
to use AnyLanguageModel with MLX support in an Xcode app project:

**1. Create a local Swift package**
(in root directory containing Xcode project):

```shell
mkdir -p Packages/MyAppKit
cd Packages/MyAppKit
swift package init
```

**2. Specify AnyLanguageModel package dependency**
(in `Packages/MyAppKit/Package.swift`):

```swift
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "MyAppKit",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "MyAppKit",
            targets: ["MyAppKit"]
        )
    ],
    dependencies: [
        .package(
            url: "https://github.com/mattt/AnyLanguageModel",
            from: "0.4.0",
            traits: ["MLX"]
        )
    ],
    targets: [
        .target(
            name: "MyAppKit",
            dependencies: [
                .product(name: "AnyLanguageModel", package: "AnyLanguageModel")
            ]
        )
    ]
)
```

**3. Export the AnyLanguageModel module**
(in `Sources/MyAppKit/Export.swift`):

```swift
@_exported import AnyLanguageModel
```

**4. Add the local package to your Xcode project**:

Open your project settings,
navigate to the "Package Dependencies" tab,
and click "+" → "Add Local..." to select the `Packages/MyAppKit` directory.

Your app can now import `AnyLanguageModel` with MLX support enabled.

> [!TIP]
> For a working example of package traits in an Xcode app project,
> see [chat-ui-swift](https://github.com/mattt/chat-ui-swift).

## API Credentials and Security

When using third-party language model providers like OpenAI, Anthropic, or Google Gemini,
you must handle API credentials securely.

> [!CAUTION]
> **Never hardcode API credentials in your app**.
> Malicious actors can reverse‑engineer your application binary
> or observe outgoing network requests
> (for example, on a compromised device or via a debugging proxy)
> to extract embedded credentials.
> There have been documented cases of attackers successfully exfiltrating
> API keys from mobile apps and racking up thousands of dollars in charges.

Here are two approaches for managing API credentials in production apps:

### Bring Your Own Key (BYO)

Users provide their own API keys,
which are stored securely in the system Keychain
and sent directly to the provider in API requests.

**Security considerations**:

- Keychain data is encrypted using hardware-backed keys
  (protected by the Secure Enclave on supported devices)
- An attacker would need access to a running process to intercept credentials
- TLS encryption protects credentials in transit on the network
- Users can only compromise their own keys, not other users' keys

**Trade-offs**:

- Apple App Review has often rejected apps using this model
- Reviewers may be unable to test functionality — even with provided credentials
- Apple may require in-app purchase integration for usage credits
- Some users may find it inconvenient to obtain and enter API keys

### Proxy Server

Instead of connecting directly to the provider,
route requests through your own authenticated service endpoint.
API credentials are stored securely on your server,
never in the client app.

Authenticate users with [OAuth 2.1](https://oauth.net/2.1/) or similar,
issuing short-lived, scoped bearer tokens for client requests.
If an attacker extracts tokens from your app,
they're limited in scope and expire automatically.

**Security considerations**:

- API keys never leave your server infrastructure
- Client tokens can be scoped
  (e.g., rate-limited, feature-restricted)
- Client tokens can be revoked or expired independently
- Compromised tokens have limited blast radius

**Trade-offs**:

- Additional infrastructure complexity
  (server, authentication, monitoring)
- Operational costs
  (hosting, maintenance, support)
- Network latency from additional hop

Fortunately, there are platforms and services that simplify proxy implementation,
handling authentication, rate limiting, and billing for you.

> [!TIP]
> For development and testing, it's fine to use API keys from environment variables.
> Just make sure production builds use one of the secure approaches above.

For more information about security best practices for your app,
see OWASP's
[Mobile Application Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Mobile_Application_Security_Cheat_Sheet.html).

## Usage

### Guided Generation

All on-device models — Apple Foundation Models, Core ML, MLX, and llama.cpp —
support guided generation,
letting you request strongly typed outputs using `@Generable` and `@Guide`
instead of parsing raw strings.
Cloud providers (OpenAI, Open Responses, Anthropic, and Gemini)
also support guided generation.
For more details, see
[Generating Swift data structures with guided generation](https://developer.apple.com/documentation/foundationmodels/generating-swift-data-structures-with-guided-generation).

```swift
@Generable(description: "Basic profile information about a cat")
struct CatProfile {
    // A guide isn't necessary for basic fields.
    var name: String

    @Guide(description: "The age of the cat", .range(0...20))
    var age: Int

    @Guide(description: "A one sentence profile about the cat's personality")
    var profile: String
}

let session = LanguageModelSession(model: model)
let response = try await session.respond(
    to: "Generate a cute rescue cat",
    generating: CatProfile.self
)
print(response.content)
```

### Image Inputs

Many providers support image inputs,
letting you include images alongside text prompts.
Pass images using the `images:` or `image:` parameter on `respond`:

```swift
let response = try await session.respond(
    to: "Describe what you see",
    images: [
        .init(url: URL(string: "https://example.com/photo.jpg")!),
        .init(url: URL(fileURLWithPath: "/path/to/local.png"))
    ]
)
```

Image support varies by provider:

| Provider                | Image Inputs    |
| ----------------------- | :-------------: |
| Apple Foundation Models | —               |
| Core ML                 | —               |
| MLX                     | model-dependent |
| llama.cpp               | —               |
| Ollama                  | model-dependent |
| OpenAI                  | yes             |
| Open Responses          | yes             |
| Anthropic               | yes             |
| Google Gemini           | yes             |

For MLX and Ollama,
use a vision-capable model 
(for example, a VLM or `-vl` variant).

### Tool Calling

Tool calling is supported by all providers except llama.cpp.
Define tools using the `Tool` protocol and pass them when creating a session:

```swift
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

let session = LanguageModelSession(model: model, tools: [WeatherTool()])

let response = try await session.respond {
    Prompt("How's the weather in Cupertino?")
}
print(response.content)
```

To observe or control tool execution, assign a delegate on the session:

```swift
actor ToolExecutionObserver: ToolExecutionDelegate {
    func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: LanguageModelSession) async {
        print("Generated tool calls: \(toolCalls)")
    }

    func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: LanguageModelSession
    ) async -> ToolExecutionDecision {
        // Return .stop to halt after tool calls, or .provideOutput(...) to bypass execution.
        // This is a good place to ask the user for confirmation (for example, in a modal dialog).
        .execute
    }

    func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: LanguageModelSession
    ) async {
        print("Executed tool call: \(toolCall)")
    }
}

session.toolExecutionDelegate = ToolExecutionObserver()
```

## Providers

### Apple Foundation Models

Uses Apple's [system language model](https://developer.apple.com/documentation/FoundationModels)
(requires macOS 26 / iOS 26 / visionOS 26 or later).

```swift
let model = SystemLanguageModel.default
let session = LanguageModelSession(model: model)

let response = try await session.respond {
    Prompt("Explain quantum computing in one sentence")
}
```

### Core ML

Runs [Core ML](https://developer.apple.com/documentation/coreml) models
(requires `CoreML` trait):

```swift
let model = CoreMLLanguageModel(url: URL(fileURLWithPath: "path/to/model.mlmodelc"))

let session = LanguageModelSession(model: model)
let response = try await session.respond {
    Prompt("Summarize this text")
}
```

Enable the trait in Package.swift:

```swift
.package(
    url: "https://github.com/mattt/AnyLanguageModel.git",
    branch: "main",
    traits: ["CoreML"]
)
```

### MLX

Runs [MLX](https://github.com/ml-explore/mlx-swift) models on Apple Silicon
(requires `MLX` trait):

```swift
let model = MLXLanguageModel(modelId: "mlx-community/Qwen3-0.6B-4bit")

let session = LanguageModelSession(model: model)
let response = try await session.respond {
    Prompt("What is the capital of France?")
}
```

Vision support depends on the specific MLX model you load.
Use a vision‑capable model for multimodal prompts
(for example, a VLM variant).
The following shows extracting text from an image:

```swift
let ocr = try await session.respond(
    to: "Extract the total amount from this receipt",
    images: [
        .init(url: URL(fileURLWithPath: "/path/to/receipt_page1.png")),
        .init(url: URL(fileURLWithPath: "/path/to/receipt_page2.png"))
    ]
)
print(ocr.content)
```

Enable the trait in Package.swift:

```swift
.package(
    url: "https://github.com/mattt/AnyLanguageModel.git",
    branch: "main",
    traits: ["MLX"]
)
```

### llama.cpp (GGUF)

Runs GGUF quantized models via [llama.cpp](https://github.com/ggml-org/llama.cpp)
(requires `Llama` trait):

```swift
let model = LlamaLanguageModel(modelPath: "/path/to/model.gguf")

let session = LanguageModelSession(model: model)
let response = try await session.respond {
    Prompt("Translate 'hello world' to Spanish")
}
```

Enable the trait in Package.swift:

```swift
.package(
    url: "https://github.com/mattt/AnyLanguageModel.git",
    branch: "main",
    traits: ["Llama"]
)
```

Configuration is done via custom generation options,
allowing you to control runtime parameters per request:

```swift
var options = GenerationOptions(temperature: 0.8)
options[custom: LlamaLanguageModel.self] = .init(
    contextSize: 4096,        // Context window size
    batchSize: 512,           // Batch size for evaluation
    threads: 8,               // Number of threads
    seed: 42,                 // Random seed for deterministic output
    temperature: 0.7,         // Sampling temperature
    topK: 40,                 // Top-K sampling
    topP: 0.95,               // Top-P (nucleus) sampling
    repeatPenalty: 1.2,       // Penalty for repeated tokens
    repeatLastN: 128,         // Number of tokens to consider for repeat penalty
    frequencyPenalty: 0.1,    // Frequency-based penalty
    presencePenalty: 0.1,     // Presence-based penalty
    mirostat: .v2(tau: 5.0, eta: 0.1)  // Adaptive perplexity control
)

let response = try await session.respond(
    to: "Write a story",
    options: options
)
```

### Ollama

Run models locally via Ollama's
[HTTP API](https://github.com/ollama/ollama/blob/main/docs/api.md):

```swift
// Default: connects to http://localhost:11434
let model = OllamaLanguageModel(model: "qwen3") // `ollama pull qwen3:8b`

// Custom endpoint
let model = OllamaLanguageModel(
    endpoint: URL(string: "http://remote-server:11434")!,
    model: "llama3.2"
)

let session = LanguageModelSession(model: model)
let response = try await session.respond {
    Prompt("Tell me a joke")
}
```

For local models, make sure you're using a vision‑capable model
(for example, a `-vl` variant).
You can combine multiple images:

```swift
let model = OllamaLanguageModel(model: "qwen3-vl") // `ollama pull qwen3-vl:8b`
let session = LanguageModelSession(model: model)
let response = try await session.respond(
    to: "Compare these posters and summarize their differences",
    images: [
        .init(url: URL(string: "https://example.com/poster1.jpg")!),
        .init(url: URL(fileURLWithPath: "/path/to/poster2.jpg"))
    ]
)
print(response.content)
```

Pass any model-specific parameters using custom generation options:

```swift
var options = GenerationOptions(temperature: 0.8)
options[custom: OllamaLanguageModel.self] = [
    "seed": .int(42),
    "repeat_penalty": .double(1.2),
    "num_ctx": .int(4096),
    "stop": .array([.string("###")])
]
```

### OpenAI

Supports both
[Chat Completions](https://platform.openai.com/docs/api-reference/chat) and
[Responses](https://platform.openai.com/docs/api-reference/responses) APIs:

```swift
let model = OpenAILanguageModel(
    apiKey: ProcessInfo.processInfo.environment["OPENAI_API_KEY"]!,
    model: "gpt-4o-mini"
)

let session = LanguageModelSession(model: model)
let response = try await session.respond(
    to: "List the objects you see",
    images: [
        .init(url: URL(string: "https://example.com/desk.jpg")!),
        .init(
            data: try Data(contentsOf: URL(fileURLWithPath: "/path/to/closeup.png")),
            mimeType: "image/png"
        )
    ]
)
print(response.content)
```

For OpenAI-compatible endpoints that use older Chat Completions API:

```swift
let model = OpenAILanguageModel(
    baseURL: URL(string: "https://api.example.com")!,
    apiKey: apiKey,
    model: "gpt-4o-mini",
    apiVariant: .chatCompletions
)
```

Use custom generation options for advanced parameters like sampling controls,
reasoning effort (for o-series models), and vendor-specific extensions:

```swift
var options = GenerationOptions(temperature: 0.8)
options[custom: OpenAILanguageModel.self] = .init(
    topP: 0.9,
    frequencyPenalty: 0.5,
    presencePenalty: 0.3,
    stopSequences: ["END"],
    reasoningEffort: .high,        // For reasoning models (o3, o4-mini)
    serviceTier: .priority,
    extraBody: [                   // Vendor-specific parameters
        "custom_param": .string("value")
    ]
)
```

### Open Responses

Connects to any API that conforms to the 
[Open Responses](https://www.openresponses.org) specification 
(e.g. OpenAI, OpenRouter, or other compatible providers). 
Base URL is required—use your provider’s endpoint:

```swift
// Example: OpenRouter (https://openrouter.ai/api/v1/)
let model = OpenResponsesLanguageModel(
    baseURL: URL(string: "https://openrouter.ai/api/v1/")!,
    apiKey: ProcessInfo.processInfo.environment["OPEN_RESPONSES_API_KEY"]!,
    model: "openai/gpt-4o-mini"
)

// Example: OpenAI
let model = OpenResponsesLanguageModel(
    baseURL: URL(string: "https://api.openai.com/v1/")!,
    apiKey: ProcessInfo.processInfo.environment["OPEN_RESPONSES_API_KEY"]!,
    model: "gpt-4o-mini"
)

let session = LanguageModelSession(model: model)
let response = try await session.respond(to: "Say hello")
```

Custom options support Open Responses–specific fields,
such as `tool_choice` (including `allowed_tools`) and `extraBody`:

```swift
var options = GenerationOptions(temperature: 0.8)
options[custom: OpenResponsesLanguageModel.self] = .init(
    toolChoice: .auto,
    allowedTools: ["getWeather"],
    reasoningEffort: .high,
    extraBody: ["custom_param": .string("value")]
)
```

### Anthropic

Uses the [Messages API](https://docs.claude.com/en/api/messages) with Claude models:

```swift
let model = AnthropicLanguageModel(
    apiKey: ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]!,
    model: "claude-sonnet-4-5-20250929"
)

let session = LanguageModelSession(model: model, tools: [WeatherTool()])
let response = try await session.respond {
    Prompt("What's the weather like in San Francisco?")
}
```

You can include images with your prompt.
You can point to remote URLs or construct from image data:

```swift
let response = try await session.respond(
    to: "Explain the key parts of this diagram",
    image: .init(
        data: try Data(contentsOf: URL(fileURLWithPath: "/path/to/diagram.png")),
        mimeType: "image/png"
    )
)
print(response.content)
```

Use custom generation options for Anthropic-specific parameters like
extended thinking, tool choice control, and sampling parameters:

```swift
var options = GenerationOptions(temperature: 0.7)
options[custom: AnthropicLanguageModel.self] = .init(
    topP: 0.9,
    topK: 40,
    stopSequences: ["END", "STOP"],
    thinking: .init(budgetTokens: 4096),  // Extended thinking
    toolChoice: .auto,                     // Tool selection control
    serviceTier: .priority
)
```

### Google Gemini

Uses the [Gemini API](https://ai.google.dev/api/generate-content) with Gemini models:

```swift
let model = GeminiLanguageModel(
    apiKey: ProcessInfo.processInfo.environment["GEMINI_API_KEY"]!,
    model: "gemini-2.5-flash"
)

let session = LanguageModelSession(model: model, tools: [WeatherTool()])
let response = try await session.respond {
    Prompt("What's the weather like in Tokyo?")
}
```

Send images with your prompt using remote or local sources:

```swift
let response = try await session.respond(
    to: "Identify the plants in this photo",
    image: .init(url: URL(string: "https://example.com/garden.jpg")!)
)
print(response.content)
```

Gemini models use an internal ["thinking process"](https://ai.google.dev/gemini-api/docs/thinking)
that improves reasoning and multi-step planning.
Configure thinking mode through custom generation options:

```swift
var options = GenerationOptions()

// Enable thinking with dynamic budget allocation
options[custom: GeminiLanguageModel.self] = .init(thinking: .dynamic)

// Or set an explicit number of tokens for its thinking budget
options[custom: GeminiLanguageModel.self] = .init(thinking: .budget(1024))

// Disable thinking (default)
options[custom: GeminiLanguageModel.self] = .init(thinking: .disabled)

let response = try await session.respond(to: "Solve this problem", options: options)
```

Gemini supports [server-side tools](https://ai.google.dev/gemini-api/docs/google-search)
that execute transparently on Google's infrastructure:

```swift
var options = GenerationOptions()
options[custom: GeminiLanguageModel.self] = .init(
    serverTools: [
        .googleSearch,
        .googleMaps(latitude: 35.6580, longitude: 139.7016)
    ]
)

let response = try await session.respond(
    to: "What coffee shops are nearby?",
    options: options
)
```

**Available server tools**:

- `.googleSearch`
  Grounds responses with real-time web information
- `.googleMaps`
  Provides location-aware responses
- `.codeExecution`
  Generates and runs Python code to solve problems
- `.urlContext`
  Fetches and analyzes content from URLs mentioned in prompts

> [!TIP]
> Gemini server tools are not available as client tools (`Tool`) for other models.

## Testing

Run the test suite to verify everything works correctly:

```bash
swift test
```

Tests for different language model backends have varying requirements:

| Backend        | Traits   | Environment Variables                               |
| -------------- | -------- | --------------------------------------------------- |
| CoreML         | `CoreML` | `HF_TOKEN`                                          |
| MLX            | `MLX`    | `HF_TOKEN`                                          |
| Llama          | `Llama`  | `LLAMA_MODEL_PATH`                                  |
| Anthropic      | —        | `ANTHROPIC_API_KEY`                                 |
| OpenAI         | —        | `OPENAI_API_KEY`                                    |
| Open Responses | —        | `OPEN_RESPONSES_API_KEY`, `OPEN_RESPONSES_BASE_URL` |
| Ollama         | —        | —                                                   |

Example setup for running multiple tests at once:

```bash
export HF_TOKEN=your_huggingface_token
export LLAMA_MODEL_PATH=/path/to/model.gguf
export ANTHROPIC_API_KEY=your_anthropic_key
export OPENAI_API_KEY=your_openai_key
export OPEN_RESPONSES_API_KEY=your_open_responses_key
export OPEN_RESPONSES_BASE_URL=https://api.openai.com/v1/

swift test --traits CoreML,Llama
```

> [!TIP]
> Tests that perform generation are skipped in CI environments (when `CI` is set).
> Override this by setting `ENABLE_COREML_TESTS=1` or `ENABLE_MLX_TESTS=1`.

> [!NOTE]
> MLX tests must be run with `xcodebuild` rather than `swift test`
> due to Metal library loading requirements.
> Since `xcodebuild` doesn't support package traits directly,
> you'll first need to update `Package.swift` to enable the MLX trait by default.
>
> ```diff
> - .default(enabledTraits: []),
> + .default(enabledTraits: ["MLX"]),
> ```
> 
> Pass environment variables with `TEST_RUNNER_` prefix:
>
> ```bash
> export TEST_RUNNER_HF_TOKEN=your_huggingface_token
> xcodebuild test \
>   -scheme AnyLanguageModel \
>   -destination 'platform=macOS' \
>   -only-testing:AnyLanguageModelTests/MLXLanguageModelTests
> ```

## License

This project is available under the MIT license.
See the LICENSE file for more info.
