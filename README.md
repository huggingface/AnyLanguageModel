# AnyLanguageModel

A Swift package that provides a drop-in replacement for
[Apple's Foundation Models framework](https://developer.apple.com/documentation/FoundationModels)
with support for custom language model providers.
All you need to do is change your import statement:

```diff
- import FoundationModels
+ import AnyLanguageModel
```

Anything AnyLanguageModel adds beyond Apple's API
is marked as an extension in its documentation.
[Differences from Foundation Models](#differences-from-foundation-models)
summarizes these extensions.

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
- [x] Apple [Private Cloud Compute](https://developer.apple.com/documentation/FoundationModels/PrivateCloudComputeLanguageModel) and any [`FoundationModels.LanguageModel`](https://developer.apple.com/documentation/FoundationModels/LanguageModel) conformer, including [Core AI](https://github.com/apple/coreai-models) models (OS 27)
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

- Swift 6.3+ (Xcode 26.4+);
  `PrivateCloudComputeLanguageModel` and `FoundationLanguageModel`
  compile only with Swift 6.4+ (Xcode 27+) and aren't available on tvOS
- iOS 17.0+ / macOS 14.0+ / tvOS 17.0+ / watchOS 10.0+ / visionOS 1.0+ / Linux / WASI
- `SystemLanguageModel` isn't available on watchOS,
  and is available on tvOS only with Swift 6.3
- On WASI, remote models need [a transport that you provide](#transports-on-wasi)
- CI builds watchOS with Xcode 27 and WASI with Swift 6.4

## Installation

Add this package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/huggingface/AnyLanguageModel", from: "1.2.0")
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
- `AsyncHTTPClient`: Sends remote models' requests with AsyncHTTPClient
  instead of `URLSession`
  (depends on `swift-server/async-http-client`)

By default, no traits are enabled.
To enable specific traits, specify them in your package's dependencies:

```swift
// In your Package.swift
dependencies: [
    .package(
        url: "https://github.com/huggingface/AnyLanguageModel.git",
        from: "1.2.0",
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
>         url: "https://github.com/huggingface/AnyLanguageModel.git",
>         from: "1.2.0",
>         traits: ["CoreML", "MLX", "Llama"]
>     ),
>     .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),           // CoreML
>     .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.32.3")), // MLX
>     .package(url: "https://github.com/mattt/llama.swift", .upToNextMajor(from: "2.10549.0")),    // Llama
> ]
> ```
>
> Include only the dependencies that correspond to the traits you enable.
> For more information, see [issue #135](https://github.com/huggingface/AnyLanguageModel/issues/135).

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
            url: "https://github.com/huggingface/AnyLanguageModel",
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
Ollama and the cloud providers (OpenAI, Open Responses, Anthropic, and Gemini)
also support guided generation,
including nested `@Generable` types and arrays of them.
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
| Apple Foundation Models | OS 27+          |
| Core ML                 | —               |
| MLX                     | model-dependent |
| llama.cpp               | model-dependent |
| Ollama                  | model-dependent |
| OpenAI                  | yes             |
| Open Responses          | yes             |
| Anthropic               | yes             |
| Google Gemini           | yes             |

For MLX and Ollama,
use a vision-capable model 
(for example, a VLM or `-vl` variant).
For llama.cpp,
pass the model's multimodal projector with `mmprojPath:`.

> [!NOTE]
> Image inputs are an AnyLanguageModel extension.
> See [Differences from Foundation Models](#differences-from-foundation-models).

### Prompt attachments

Image attachments use the Foundation Models 27 prompt construction syntax,
including ordinary Tool outputs:

```swift
func call(arguments: Arguments) async throws -> Prompt {
    Prompt {
        "Image read from the selected file"
        Attachment(imageURL: fileURL)
    }
}
```

On Apple platforms, `Attachment(CGImage, orientation:)` retains typed image
content and orientation. Prompts, builders, and Tool outputs preserve the order
of text and images, and the OS 26 compatibility transcript continues to store
images using its existing `Transcript.Segment.image` Codable representation.

`SystemLanguageModel` and `FoundationLanguageModel` bridge prompt attachments
to Foundation Models only in apps built with Swift 6.4 or later
that run on macOS, iOS, or visionOS 27 or later.
On earlier runtimes, and on tvOS and watchOS,
they throw an error when a prompt has an attachment.
An image in a Tool output follows the same conditions:
it goes to Foundation Models as an attachment,
and otherwise the tool call throws the same error.
The system bridge requires local file URLs; other provider adapters retain URL
images in their existing transcript representation.

Ordinary Tool outputs read local image files once when the output is created,
and retain the image bytes and MIME type in the compatibility transcript.
Continuations and saved transcript replay therefore do not depend on the original
file remaining at its URL. Remote image URLs are retained without downloading.

Gemini supports images in Tool function responses only with Gemini 3 and later.
In Gemini Tool outputs, remote image URLs are sent as text at their original
position without downloading the image; the transcript retains the image segment.

In Tool outputs, an image type that the API doesn't take is sent as `<image>` text:
Anthropic takes JPEG, PNG, GIF, and WebP,
and Gemini takes PNG, JPEG, and WebP.
In an Anthropic prompt, a local file with another extension throws an error.
Ollama takes only image data,
so it sends any image URL in a prompt, and a remote URL in a Tool output,
as text without downloading it.

This compatibility surface covers `CGImage` and `imageURL` attachments in
prompts and ordinary Tool outputs. Instruction attachments, labels, `CIImage`,
pixel buffers, and newer data attachments remain outside this change.

### Tool Calling

Tool calling is supported by all providers except Core ML.
For llama.cpp, it depends on the model's chat format.
`CoreMLLanguageModel` can render tool definitions into its chat template
with a `chatTemplateHandler` and a `toolsHandler`,
but nothing parses or runs a tool call.
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

When a delegate returns `.stop`,
the response ends with the tool calls that stopped it.
`respond` returns an empty value,
even if the model wrote text before the tool calls:
`""` for `String`, and otherwise the first of `{}`, `null`, and `[]`
that decodes as the type.
After a profile hands the response off to another model,
a `String` response starts with the text from before the hand-off.
A stream keeps its last snapshot if that snapshot decodes as the complete type,
so a `String` stream ends with the text so far.
Otherwise, a stream also ends with an empty value.
Structured streams from MLX and llama.cpp follow the rule for `respond`.
A type that can't be empty, such as one with required properties,
a number, or a Boolean,
throws `LanguageModelSession.ToolExecutionStopped` instead,
with the rounds that ran and the stopped calls in its `transcriptEntries`.

A delegate can also steer a response while it runs.
After each round of tool calls, before the model's next request,
the session asks the delegate for messages to add,
and the model reads them after the round's tool outputs:

```swift
actor Steering: ToolExecutionDelegate {
    private var inbox: [Prompt] = []

    func send(_ message: Prompt) { inbox.append(message) }

    func pendingMessages(in session: LanguageModelSession) async -> [Prompt] {
        defer { inbox.removeAll() }
        return inbox
    }
}
```

Each message joins the transcript as a prompt, after the round's tool outputs,
and is sent in the request in the same form, so later responses replay it as it was sent.
Messages still waiting when a response ends without another round of tool calls
stay with your app, to send as the next prompt.
`SystemLanguageModel` doesn't ask, because Foundation Models runs its own tool loop.

> [!NOTE]
> Tool execution delegates are an AnyLanguageModel extension.
> See [Differences from Foundation Models](#differences-from-foundation-models).

### Dynamic Instructions

`DynamicInstructions` lets a session change the instructions and tools
for each request to the model,
without creating a new session:

```swift
final class CurrentAppState {
    var canCheckWeather = false
}

struct CurrentAppInstructions: DynamicInstructions {
    let state: CurrentAppState

    var body: some DynamicInstructions {
        Instructions("Help with the currently visible app.")
        if state.canCheckWeather {
            WeatherTool()
        }
    }
}

let state = CurrentAppState()
let session = LanguageModelSession(
    model: model,
    dynamicInstructions: CurrentAppInstructions(state: state),
    history: savedHistory
)
```

The session evaluates the body before every request to the model,
including the request that continues a response after tool calls.
The resolved instructions are sent with each request
but never become part of the session's transcript.

> [!NOTE]
> Dynamic instructions follow the Foundation Models 27 API.
> `SystemLanguageModel` supports them only in apps built with Swift 6.4 or later
> that run on OS 27 or later, and not on tvOS.
> Otherwise, it throws `SystemLanguageModel.Error.dynamicInstructionsUnavailable`
> for a session with dynamic instructions.

### Foundation Models 27 request options

`GenerationOptions` has the Foundation Models 27 names
`samplingMode` and `init(samplingMode:temperature:maximumResponseTokens:)`,
which Foundation Models back-deploys to OS 26.
The older `sampling` and `init(sampling:temperature:maximumResponseTokens:)`
remain available.
`GenerationOptions` also has a `toolCallingMode`,
and `ContextOptions` holds the schema-in-prompt and reasoning-level settings
from Foundation Models 27:

```swift
let options = GenerationOptions(
    samplingMode: .greedy,
    toolCallingMode: .required
)
let context = ContextOptions(
    includeSchemaInPrompt: true,
    reasoningLevel: .moderate
)
```

A `nil` tool-calling mode, the default, leaves each model's behavior unchanged.
Each provider handles the three modes as follows:

| Provider | `allowed` | `disallowed` | `required` |
| --- | --- | --- | --- |
| `SystemLanguageModel`, `PrivateCloudComputeLanguageModel`, `FoundationLanguageModel` | Passed to Foundation Models | Passed to Foundation Models | Passed to Foundation Models |
| `OpenAILanguageModel` (both APIs), `OpenResponsesLanguageModel` | `"tool_choice": "auto"` | `"tool_choice": "none"` | `"tool_choice": "required"` |
| `AnthropicLanguageModel` | `"tool_choice": {"type": "auto"}` | `"tool_choice": {"type": "none"}` | `"tool_choice": {"type": "any"}` |
| `GeminiLanguageModel` | `functionCallingConfig` mode `AUTO` | `functionCallingConfig` mode `NONE` | `functionCallingConfig` mode `ANY` |
| `OllamaLanguageModel`, `MLXLanguageModel`, `LlamaLanguageModel` | Tools offered as usual | No tools offered | Not supported: tools offered as usual |
| `CoreMLLanguageModel` | Tool definitions rendered as usual; no tool runs | No tool definitions rendered | Not supported: tool definitions rendered as usual; no tool runs |

The models that run through Foundation Models pass the mode on
in apps built with Swift 6.4 or later that run on OS 27 or later,
and not on tvOS.
Otherwise, they ignore it.

The remote providers send the mode only with a request that has tools,
because a tool choice applies to the tools in the request.
For Gemini, server tools such as Google Search don't count:
the mode applies to the session's tools.
A provider's own tool-choice option takes precedence over `toolCallingMode`:
`toolChoice` in the custom options of `AnthropicLanguageModel`
and `OpenResponsesLanguageModel`,
and a `tool_choice` key in `extraBody`
for `OpenAILanguageModel`, `OpenResponsesLanguageModel`, and `AnthropicLanguageModel`.

The `toolChoice` cases that match a tool-calling mode are deprecated;
set `toolCallingMode` instead:

| `toolCallingMode` | Deprecated `AnthropicLanguageModel` case | Deprecated `OpenResponsesLanguageModel` case |
| --- | --- | --- |
| `.allowed` | `.auto` | `.auto` |
| `.required` | `.any` | `.required` |
| `.disallowed` | `.disabled` | `.none` |

The deprecated cases still send the same requests.
To make the model call a specific tool,
or to restrict it to some of the tools,
use the cases that remain:
`.tool(name:)` for `AnthropicLanguageModel`,
and `.function(name:)` or `.allowedTools(tools:mode:)` for `OpenResponsesLanguageModel`.

Ollama, MLX, llama.cpp, and Core ML have no tool-choice setting.
They honor `disallowed` by leaving the tools out of the prompt,
so no tool runs.
They can't make a model call a tool,
so they treat `required` like `allowed`.

As in Foundation Models, the mode applies to every request in a response,
including each request that continues the response after tool calls.
With `required`, the model keeps calling tools
until a tool throws an error,
a `ToolExecutionDelegate` stops the response,
dynamic instructions remove the tools,
or a round of tool calls past `maximumToolRounds`
ends the response with `LanguageModelSession.ToolRoundLimitExceeded`.
`SystemLanguageModel`, `PrivateCloudComputeLanguageModel`, and `FoundationLanguageModel`
ignore `maximumToolRounds`,
so with them, end the loop with a delegate's `.stop`
or with dynamic instructions that remove the tools.
Without tools, a `required` request to these models throws a Foundation Models error.

> [!NOTE]
> Anthropic rejects `{"type": "any"}`
> with manual extended thinking (`thinking: .enabled(budgetTokens:)`)
> and on models that don't support forced tool use.
> See [Forcing tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use).

A transcript keeps the tool-calling mode of each prompt.

No AnyLanguageModel API accepts `ContextOptions` yet,
so creating one has no effect on a request.

### Dynamic Profiles

`LanguageModelSession.DynamicProfile` mirrors the Foundation Models 27 profile
API on supported AnyLanguageModel deployment targets. A profile chooses one
active `Profile` for each actual model request and can change its model,
instructions, tools, generation options, and transcript
projection without rebuilding the session:

```swift
extension SessionPropertyValues {
    @SessionPropertyEntry var isReviewing = false
}

struct EditingProfile: LanguageModelSession.DynamicProfile {
    let model: any LanguageModel
    @SessionProperty(\.isReviewing) private var isReviewing

    var body: some LanguageModelSession.DynamicProfile {
        if isReviewing {
            Profile {
                Instructions("Review the current draft.")
                ReviewTool()
            }
            .model(model)
            .temperature(0.1)
        } else {
            Profile {
                Instructions("Help edit the current draft.")
                EditTool()
            }
            .model(model)
            .temperature(0.7)
        }
    }
}

let session = LanguageModelSession(
    profile: EditingProfile(model: model),
    history: savedHistory
)
session.properties.isReviewing = true
```

A profile must set a model with `.model(_:)`.
Unlike Foundation Models, which falls back to the system language model,
a request whose active profile sets no model
throws `LanguageModelSession.DynamicProfileError.missingModel`.
`init(profile:history:)` keeps a first instructions entry in `history`
as the base instructions and leaves out any other instructions entries.

Value modifiers use call-site, inner-profile, then outer-profile precedence:
a non-`nil` generation option passed with a request overrides the profile,
and otherwise the innermost modifier, closest to the `Profile`, sets the value.
Lifecycle callbacks accumulate, and the outermost runs first,
except for `onDeactivate`, where the innermost runs first.
History transforms also run outermost first,
and each one receives the entries that the one before it returned.
When the active profile changes,
the previous profile's `onDeactivate` actions run,
then the new profile's `onActivate` actions,
and then `onPrompt` actions run before the first model request for the prompt.
`onReasoning` actions run before `onResponse` actions.

A session runs its profile's callbacks and lifecycle actions one at a time,
even when requests overlap.
While a callback waits for a request or a response stream,
other callbacks can run, so captured state can change across that wait.
A callback that waits for another task
that makes a request on the same session never finishes.

`historyTransform` changes only the transcript
projection for that request; `session.transcript` remains the canonical history,
unless the transform writes to the `history` session property.

When a turn is replayed on OpenAI (both API variants), Open Responses, and Gemini,
edits to a string response take precedence
over the text the session stored when it wrote the turn:
a changed or removed response sends only its current text and the turn's calls.
Structured responses are the limit.
Editing or removing one doesn't remove text from that turn's earlier tool rounds,
which stays in internal metadata on the `toolCalls` entry and is still sent.
During streaming, a caller may have seen that text in snapshots;
with `respond`, it never appears.
To remove it, drop or rebuild the `toolCalls` entry.
The same limit applies when a transform changes a prompt
so that a string turn counts as structured,
and it can apply when a transform removes a later prompt
but keeps its `toolCalls` entry.

An edit in place also leaves the original text in the metadata of the entry
or the response in a saved transcript.
For a string turn, that text isn't sent, but it's still stored.
A plain `historyTransform` never changes the stored transcript.
To remove text from it, write `history` with those entries dropped or rebuilt.

`@SessionProperty` exposes isolated session state to profiles, dynamic
instructions, and tools. The built-in `history` property is writable from a
profile or lifecycle callback and read-only while dynamic instructions or a Tool
is active.
Writing `history` from dynamic instructions or a Tool's `call` stops the process.
So does reading or writing a `@SessionProperty` anywhere except
a session's profile, including its callbacks and history transforms,
its dynamic instructions, and a Tool's `call` during its response.
For example, calling a Tool's `call` directly,
reading a property in a Tool's `description`,
and reading one in a `ToolExecutionDelegate` all stop the process.
Use `session.properties` in those places.

The five HTTP providers (OpenAI, Open Responses, Anthropic, Gemini, and Ollama)
evaluate the profile and its history transforms again
before each request that continues a response after tool calls,
and run `onToolCall` and `onToolOutput`.
With these providers, a Tool's `history` property includes
the tool calls and outputs of the current response, including the Tool's own call.

Create a profile's model outside `body`, as `EditingProfile` does.
Each model that these providers create has its own identity, which copies share,
so `.model(OpenAILanguageModel(apiKey: key, model: name))` in `body`
selects a different model on each evaluation,
and each continuation hands the response off to the new one.
With the OpenAI Responses API and Open Responses,
the next request then leaves out the reasoning item and the function call's `id`.

MLX, llama.cpp, Core ML, Private Cloud Compute, `FoundationLanguageModel`,
and the system model don't run `onToolCall` or `onToolOutput` yet.
MLX and llama.cpp evaluate the profile once for each response,
so their tool rounds use the same instructions, tools, and options,
and a Tool's `history` property doesn't include the response's tool rounds.

The system model, Private Cloud Compute, and `FoundationLanguageModel`
run tools in a native Foundation Models session.
They evaluate the profile and its history transforms again
before the continuation after a tool round,
so the continuation uses the new instructions, tools, and transcript projection.
A continuation that selects another model stays on the native session's model,
and changed options don't apply to it.
A Tool's `history` property doesn't include the Tool's own call.
As with dynamic instructions, the system model supports profiles
only in apps built with Swift 6.4 or later
that run on OS 27 or later, and not on tvOS.

External compatibility models that own Tool continuation use
`resolvedRequestContext(including:options:)`,
which the `Compatibility` SPI exposes, before every provider request. The
`including:` entries are the canonical Tool calls and outputs completed in the
current response but not yet committed to the session transcript. Calls execute
against the tools in the producing request context; the next resolved context is
used only for the continuation request.

### Reasoning in the transcript

Reasoning is transcript content, separate from the answer in `response.content`.
A provider can emit `Transcript.Entry.reasoning` through the existing cumulative
`transcriptEntries` on responses and streaming snapshots. Each `Transcript.Reasoning`
contains a stable `id`, display `segments`, opaque `signature: Data?`, and metadata.
Treat successive snapshots as updates to the same entries, not new history rows.

Three built-in providers populate these entries
in streaming and nonstreaming responses,
and send them back to the model in later tool rounds
and when a session replays its transcript:

- `AnthropicLanguageModel`, for thinking and redacted thinking.
- `OllamaLanguageModel`, for a model's `thinking`.
- `OpenAILanguageModel` with the Chat Completions API,
  for `reasoning_content` (DeepSeek)
  or `reasoning` and `reasoning_details` (OpenRouter and Vercel AI Gateway).
  Hosts that serve Chat Completions don't agree on these fields,
  so an entry goes back only in requests to the host that wrote it.

For example, with Anthropic:

```swift
let model = AnthropicLanguageModel(apiKey: apiKey, model: modelID)
let session = LanguageModelSession(model: model)
var options = GenerationOptions(maximumResponseTokens: 4096)
options[custom: AnthropicLanguageModel.self] = .init(thinking: .init(budgetTokens: 1024))
for try await snapshot in session.streamResponse(to: "Explain your approach", options: options) {
    let reasoning = snapshot.transcriptEntries.compactMap { entry -> String? in
        guard case .reasoning(let value) = entry else { return nil }
        return value.segments.compactMap { segment -> String? in
            guard case .text(let text) = segment else { return nil }
            return text.content
        }.joined()
    }.joined()
    // Replace the displayed reasoning and answer independently.
    print(reasoning)
    print(snapshot.content)
}
let savedTranscript = try JSONEncoder().encode(session.transcript)
```

Choose an Anthropic model and thinking budget that support this configuration.
Redacted thinking has no display segments. Signatures and metadata are opaque
replay state; preserve them with the transcript, and do not display them as text.
Each of these providers replays only its own reasoning entries,
including after Codable restoration,
and skips reasoning from other providers.
Providers that can't replay reasoning omit those entries from their requests.
The original reasoning remains in the transcript for display and persistence.
Anthropic still validates its own replay signatures. CoreML keeps its existing
prompt-only behavior and does not send transcript history. For structured scalar outputs that
cannot represent an absent partial value, reasoning updates wait until a valid
partial answer is available. Cancellation behavior is unchanged.

### Token Usage

Inspect token counts with `response.usage`
and track accumulated usage with `session.usage`:

```swift
let session = LanguageModelSession(model: model)
let response = try await session.respond(to: "Explain how rainbows form.")
let usage = response.usage

print("Input tokens:", usage.input.totalTokenCount)
print("Cached input tokens:", usage.input.cachedTokenCount)
print("Output tokens:", usage.output.totalTokenCount)
print("Reasoning tokens:", usage.output.reasoningTokenCount)
print("Total tokens:", usage.totalTokenCount)
print("Session total:", session.usage.totalTokenCount)
```

OpenAI (Chat Completions and Responses),
Open Responses, Anthropic, Google Gemini, Ollama,
MLX, llama.cpp, and Core ML report token usage.
`SystemLanguageModel`, `PrivateCloudComputeLanguageModel`, and `FoundationLanguageModel`
also report it in apps built with Swift 6.4 or later
that run on OS 27 or later, and not on tvOS.
Counts follow each provider's definitions.
A response that calls tools makes a request for each round.
As in Foundation Models, `response.usage` reports the final request,
and `session.usage` counts every request.
Usage and individual counts are non-optional;
counts that a provider doesn't report default to zero.

Streaming snapshots carry the latest reported counts for the current request,
which may arrive after the last text update.
MLX and llama.cpp report totals after each generation round;
Core ML updates counts as token sequences arrive
and reconciles them with the final returned sequence.
Local adapters count prepared prompt tokens and generated tokens,
including tool-call output and forced JSON syntax for structured generation.
MLX and llama.cpp include reused prompt tokens in the input total
and report the prefix actually reused as cached input.
Core ML reports zero cached tokens because it resets model state.
Multimodal counts follow the runtime's prepared input definition:
llama.cpp includes native image-chunk tokens,
and MLX uses its prepared input-token count without estimating extra image costs.
Structured streams from MLX, llama.cpp, and Core ML
yield one completed snapshot with usage.
`collect()` preserves the final snapshot's usage.
Session usage increases as responses and snapshots report counts,
without counting the same tokens more than once.
Reported counts remain included if a stream later fails.
A session restored from a transcript starts with zero accumulated usage.

Models can provide additional statistics in `usage.metadata`,
a dictionary of `GeneratedContent` values.
The initializer accepts values that conform to `ConvertibleToGeneratedContent`.
When accumulating session usage,
the latest value is kept for each metadata key.

> [!NOTE]
> Token usage extends the Foundation Models 26 API
> and follows the documented
> [Foundation Models 27 usage API](https://developer.apple.com/documentation/foundationmodels/languagemodelsession/usage-swift.struct).
> `Codable` and `Equatable` support are AnyLanguageModel extensions.

### HTTP Transport

The remote models
(OpenAI, Open Responses, Anthropic, Google Gemini, and Ollama)
send each HTTP request through a transport.
Pass your own transport as `transport:` when you create a model
to change how its requests are sent:
for example, to use custom network settings or a proxy,
to log or limit requests,
or to answer requests in tests without a network connection.

By default, a model uses `makeDefaultTransport()`,
which returns a `URLSession` with the default configuration.
With the `AsyncHTTPClient` [trait](#package-traits),
it returns AsyncHTTPClient's shared `HTTPClient` instead.
On WASI, there's no default transport,
so instead provide a remote model one of your own
([see below](#transports-on-wasi)).
`URLSession` and `HTTPClient` are both transports,
so you can pass one that you configured:

```swift
let configuration = URLSessionConfiguration.default
configuration.timeoutIntervalForRequest = 600 // Give a large model time to load

let model = OllamaLanguageModel(
    model: "qwen3.5",
    transport: URLSession(configuration: configuration)
)
```

To write your own transport,
conform a type to `HTTPTransport`
and implement `send(request:body:timeout:responseHandler:)`.
The transport sends the request
and gives the response to `responseHandler`,
which reads the body as it arrives.
The request ends when the handler returns.
This transport prints each request and its response status,
then passes the request to another transport:

```swift
import AnyLanguageModel
import Foundation
import HTTPTypes

struct LoggingTransport: HTTPTransport {
    var base: any HTTPTransport = makeDefaultTransport()

    func send<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration?,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return {
        print(request.method, request.path ?? "/")
        return try await base.send(request: request, body: body, timeout: timeout) { response, body in
            print(response.status)
            return try await responseHandler(response, body)
        }
    }
}

let model = OpenAILanguageModel(
    apiKey: ProcessInfo.processInfo.environment["OPENAI_API_KEY"]!,
    model: "gpt-5.6-luna",
    transport: LoggingTransport()
)
```

`HTTPRequest` and `HTTPResponse` come from Apple's
[swift-http-types](https://github.com/apple/swift-http-types) package.
To import `HTTPTypes`,
add that package's `HTTPTypes` product to your target's dependencies.

A transport gets each HTTP request, not each call to `respond`.
A response that calls tools makes a request for each round,
so a transport that limits concurrent requests counts each round separately.
The documentation comment for `send(request:body:timeout:responseHandler:)`
gives the rules that every transport must obey,
such as how to handle cancellation and when not to retry a request.

In tests, a fake transport can answer each request with a fixed response:

```swift
struct FixedResponseTransport: HTTPTransport {
    var json: String

    func send<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration?,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return {
        let chunks = AsyncStream<Data> { continuation in
            continuation.yield(Data(json.utf8))
            continuation.finish()
        }
        return try await responseHandler(HTTPResponse(status: .ok), HTTPResponseBody(chunks))
    }
}

let model = OllamaLanguageModel(
    model: "qwen3.5",
    transport: FixedResponseTransport(json: """
        {"model": "qwen3.5", "created_at": "2026-01-01T00:00:00Z",
         "message": {"role": "assistant", "content": "Hello"}, "done": true}
        """)
)
let response = try await LanguageModelSession(model: model).respond(to: "Hi")
print(response.content) // Hello
```

When the provider answers with an error status,
the model throws an `HTTPTransportError`
with the response's status, headers, and body,
so you can read a header such as `Retry-After`.
Known provider error events in an HTTP 200 SSE stream also terminate the stream.
They throw `HTTPTransportError` with `kind == .providerError`,
the successful HTTP response, and the failed event's JSON data in `body`.
Network failures throw the transport's own error,
such as `URLError` or AsyncHTTPClient's `HTTPClientError`.

Streamed requests give the model's `responseStartTimeout` to the transport
as the `timeout` of `send(request:body:timeout:responseHandler:)`.
Set it to give a model more time to start its response,
such as a local model that loads before its first token,
or to fail sooner:

```swift
let model = OllamaLanguageModel(
    model: "qwen3.5",
    responseStartTimeout: .seconds(300)
)
```

The default is `nil`, and requests that don't stream always pass `nil`.
With `nil`, the transport decides:
a `URLSession` uses its configuration's `timeoutIntervalForRequest`,
and AsyncHTTPClient waits up to 180 seconds for the response to start.
The two transports apply a timeout differently.
AsyncHTTPClient applies it only until the response starts.
A `URLSession` applies it to each pause in the response,
so it also limits the wait between parts of a stream.
On Linux, a `URLSession` counts only whole seconds,
so it rounds a timeout up to the next second.
Both transports reduce a very large timeout to a value that they can hold,
so a value such as `.seconds(Int64.max)` sets no practical limit
and doesn't cause a crash.
A custom transport decides when its timeout starts,
so a request that waits for a free slot before it's sent
doesn't use its time while it waits.

#### Transports on WASI

WASI has no `URLSession` or AsyncHTTPClient,
so there's no default transport there:
a remote model created without a `transport:`
throws an `HTTPTransportError` when it sends a request.
Instead, write a transport that sends requests through your host.
For example, in a browser or on Cloudflare Workers,
a transport can call JavaScript's `fetch`
with [JavaScriptKit](https://github.com/swiftwasm/JavaScriptKit).
This one reads each response whole rather than streaming it:

```swift
import AnyLanguageModel
import Foundation
import HTTPTypes
import JavaScriptEventLoop
import JavaScriptKit

struct FetchTransport: HTTPTransport {
    func send<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration?,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return {
        guard let url = request.url else {
            throw HTTPTransportError(kind: .invalidRequest)
        }

        let headers = JSObject()

        for field in request.headerFields {
            headers[field.name.rawName] = .string(field.value)
        }
        
        let options = JSObject()
        options.method = .string(request.method.rawValue)
        options.headers = .object(headers)
       
        if let body {
            options.body = .string(String(decoding: body, as: UTF8.self))
        }

        let controller = JSObject.global.AbortController.function!.new()
        options.signal = controller.signal
        // `JSObject` isn't `Sendable`, so the cancellation handler
        // reaches the controller through a `JSRemote`.
        let remoteController = JSRemote(controller)

        // Abort the `fetch` if its response doesn't start within `timeout`.
        let timer: JSValue? = timeout.map { timeout in
            let (seconds, attoseconds) = timeout.components
            let milliseconds = Double(seconds) * 1_000 + Double(attoseconds) / 1e15
            return JSObject.global.setTimeout!(controller.abort.function!.bind!(controller), milliseconds)
        }
        func clearTimer() {
            if let timer {
                _ = JSObject.global.clearTimeout!(timer)
            }
        }
        // Clear the timer when `fetch` fails or is cancelled, too.
        defer { clearTimer() }

        let (response, text) = try await withTaskCancellationHandler {
            let fetch = JSObject.global.fetch.function!
            let response = try await JSPromise(fetch(url.absoluteString, options).object!)!.value.object!
            clearTimer()
            let text = try await JSPromise(response.text!().object!)!.value.string!
            return (response, text)
        } onCancel: {
            Task { await remoteController.withJSObject { _ = $0.abort!() } }
        }

        try Task.checkCancellation()

        var headerFields = HTTPFields()
        let entries = JSObject.global.Object.function!.fromEntries!(response.headers)
    
        for (name, value) in [String: String].construct(from: entries) ?? [:] {
            if let name = HTTPField.Name(name) {
                headerFields.append(HTTPField(name: name, value: value))
            }
        }

        let (chunks, continuation) = AsyncStream.makeStream(of: Data.self)
       
        continuation.yield(Data(text.utf8))
        continuation.finish()
       
        let status = HTTPResponse.Status(code: Int(response.status.number!))
       
        return try await responseHandler(
            HTTPResponse(status: status, headerFields: headerFields),
            HTTPResponseBody(chunks)
        )
    }
}

JavaScriptEventLoop.installGlobalExecutor()

let model = AnthropicLanguageModel(
    apiKey: apiKey,
    model: "claude-haiku-5-5",
    transport: FetchTransport()
)
```

Cancelling the task that sends a request aborts the `fetch`,
so `send` throws instead of waiting for the response.
So does a response that doesn't start within the request's `timeout`.
Awaiting a `JSPromise` needs JavaScriptKit's event loop,
so install it once when your program starts.
To build for WASI,
use the [Swift SDK for WebAssembly](https://www.swift.org/documentation/articles/wasm-getting-started.html).

## Differences from Foundation Models

Code that uses an extension doesn't compile with Foundation Models on OS 26.
Each extension's documentation comment says so,
and extensions that follow a Foundation Models 27 API,
such as token usage,
say which API they follow.

### Extensions

- `LanguageModel` and `Availability`:
  the protocol that lets a session use any model provider.
  Foundation Models 27 adds its own `LanguageModel` protocol
  with different requirements.
  Session initializers require a `model:` argument
  instead of defaulting to `SystemLanguageModel.default`.
- `GenerationOptions[custom:]` and `CustomGenerationOptions`:
  options for one provider,
  described in that provider's section under [Providers](#providers).
- `respond(to:images:)`, `Transcript.ImageSegment`, and `Transcript.Segment.image`:
  [image inputs](#image-inputs).
  Foundation Models 27-style [prompt attachments](#prompt-attachments) are also
  available for ordered text and image prompt construction.
- `Tool.makeOutputSegments(from:)`, exposed only through the `Compatibility` SPI:
  an integration extension for back-deployment provider adapters to lower normal
  Tool outputs into the compatibility transcript. It is not an Apple Tool API;
  OS 27 canonical executors use Foundation Models' own Tool execution instead.
- `ToolExecutionDelegate`, `ToolExecutionDecision`, and `toolExecutionDelegate`:
  [observing and controlling tool calls](#tool-calling).
- `transcriptErrorHandlingPolicy`, which follows the Foundation Models 27 API,
  and `waitForResponseCompletion()`:
  what a transcript keeps when a request fails or is cancelled.
- `LanguageModelSession.tools`, `instructions`, and `resolvedRequestContext()`:
  the session's tools, instructions, and the inputs for each request,
  for language models defined outside AnyLanguageModel.
- `Usage` and the `usage` properties:
  [token usage](#token-usage),
  which follows the Foundation Models 27 API.
- `DynamicInstructions`, its builder, and `LanguageModelSession.init(model:dynamicInstructions:history:)`:
  [dynamic instructions](#dynamic-instructions),
  which follow the Foundation Models 27 API.
- `ContextOptions`, `GenerationOptions.ToolCallingMode`, and `GenerationOptions.toolCallingMode`:
  [request options](#foundation-models-27-request-options),
  which follow the Foundation Models 27 API.
- `LanguageModelSession.DynamicProfile`, profile modifiers, lifecycle callbacks,
  session properties, and `LanguageModelSession.init(profile:history:)`:
  [dynamic profiles](#dynamic-profiles), which mirror Foundation Models 27 on
  earlier deployment targets, but require a model.
  `resolvedRequestContext(including:options:)` and
  the explicit compatibility lifecycle hooks are AnyLanguageModel extensions for
  external model implementations, exposed only through the `Compatibility` SPI.
- `Transcript.Entry.reasoning` and `Transcript.Reasoning`:
  [reasoning in the transcript](#reasoning-in-the-transcript),
  which follows the Foundation Models 27 API.
- `GenerationOptions.maximumToolRounds` and `LanguageModelSession.ToolRoundLimitExceeded`:
  the most rounds of tool calls one response may take,
  and the error that ends a response past it.
  `SystemLanguageModel`, `PrivateCloudComputeLanguageModel`, and `FoundationLanguageModel`
  ignore the limit, so end their tool loops
  [another way](#foundation-models-27-request-options).
- `responseStartTimeout` on remote models:
  how long a [streamed request](#http-transport) waits for its response to start.
- Public initializers for `Response`, `ResponseStream`, `ResponseStream.Snapshot`,
  `GenerationGuide`, and `LanguageModelFeedback`,
  for language models defined outside AnyLanguageModel.
- `Codable` conformance for `GeneratedContent`, `GenerationID`, `Usage`,
  and the types nested in `Transcript`.
- `HTTPTransport`, `HTTPResponseBody`, `HTTPTransportError`, and `makeDefaultTransport()`:
  [HTTP transports](#http-transport) for remote models.
- `GeneratedContentError`, `Transcript.ReasoningReplayError`, `SystemLanguageModel.Error`,
  and each provider's error type.
- `JSONValue`:
  JSON values for provider options such as `extraBody`.
- Smaller additions to existing types,
  such as `LanguageModel.isAvailable`, `SystemLanguageModel.supportsImageInput`,
  `Transcript.ToolCall.providerMetadata`, and `GenerationError.Refusal.transcriptEntries`.

### Behavior differences

- A session adds the prompt to its transcript before the model responds,
  preserving its text and image segment order.
- A structured response is recorded in the transcript as JSON text,
  not as a structured segment.
- A session created from a transcript restores `instructions`
  only when the first entry is instructions with a single text segment.
- Writing the `history` session property from dynamic instructions or a Tool's `call`
  stops the process with a precondition failure.
  Foundation Models 27 ignores the write.
- Reading or writing a `@SessionProperty` outside the scopes
  that [Dynamic Profiles](#dynamic-profiles) lists stops the process.
  Use `session.properties` there.
  In Foundation Models 27, the read returns the key's default value,
  and the write has no effect.

## Providers

### Apple Foundation Models

Uses Apple's [system language model](https://developer.apple.com/documentation/FoundationModels)
(requires macOS 26 / iOS 26 / visionOS 26 or later).
Builds with Swift 6.4 or later leave `SystemLanguageModel` out on tvOS,
where the SDK marks it unavailable.
So where this README describes what the model does on tvOS,
such as throwing an error, that applies only to builds with Swift 6.3.

```swift
let model = SystemLanguageModel.default
let session = LanguageModelSession(model: model)

let response = try await session.respond {
    Prompt("Explain quantum computing in one sentence")
}
```

### Apple Private Cloud Compute

Uses Apple's [Private Cloud Compute model](https://developer.apple.com/documentation/FoundationModels/PrivateCloudComputeLanguageModel),
a larger server-hosted model behind the same privacy architecture as the on-device one
(requires macOS 27 / iOS 27 or later and the Private Cloud Compute entitlement).

```swift
let model = PrivateCloudComputeLanguageModel.default
let session = LanguageModelSession(model: model)

let response = try await session.respond {
    Prompt("Summarize the attached report in three bullet points")
}
```

### Any Foundation Models Conformer

On OS 27, Foundation Models accepts any type that conforms to its `LanguageModel` protocol.
`FoundationLanguageModel` wraps such a model so it works with everything in this package.
Construct the model yourself,
or hand the wrapper an async factory
so an expensive load happens on the first request
and you control when it is released.

```swift
let model = FoundationLanguageModel {
    try await MyModel(resourcesAt: url)
}
let session = LanguageModelSession(model: model)

let response = try await session.respond(to: "Hello")
await model.unload()
```

#### Core AI Models

Apple's [coreai-models](https://github.com/apple/coreai-models) package
exports language models for the Core AI engine,
and its `CoreAILanguageModel` conforms to the Foundation Models protocol.
Add the `CoreAILM` product from that package to your app and wrap the model:

```swift
import CoreAILanguageModels

let model = FoundationLanguageModel {
    try await CoreAILanguageModels.CoreAILanguageModel(resourcesAt: resourcesURL)
}
let session = LanguageModelSession(model: model)
```

`coreai-models` requires a deployment target of OS 27,
so you can add it only to an app that already requires OS 27.
If your app supports earlier releases,
[coreai-models-xcframework](https://github.com/james-333i/coreai-models-xcframework)
is a community recipe for shipping the package as prebuilt frameworks
with the minimum OS lowered.
The recipe isn't maintained here,
and its README covers what can break.

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
    url: "https://github.com/huggingface/AnyLanguageModel.git",
    from: "1.2.0",
    traits: ["CoreML"]
)
```

### MLX

Runs [MLX](https://github.com/ml-explore/mlx-swift) models on Apple Silicon
(requires `MLX` trait):

```swift
let model = MLXLanguageModel(modelId: "mlx-community/Qwen3.5-4B-MLX-4bit")

let session = LanguageModelSession(model: model)
let response = try await session.respond {
    Prompt("What is the capital of France?")
}
```

You can tune MLX request behavior per call with model-specific options,
including sampling, KV-cache settings, and optional media preprocessing:

```swift
var options = GenerationOptions(temperature: 0.7)
var mlxOptions = MLXLanguageModel.CustomGenerationOptions.default
mlxOptions.kvCache = .init(
    maxSize: 4096,
    bits: 4,
    groupSize: 64,
    quantizedStart: 128
)
// Apply a deterministic preprocessing step for image inputs.
mlxOptions.userInputProcessing = .resize(to: CGSize(width: 512, height: 512))
// Inject extra template context consumed by model-specific chat templates.
mlxOptions.additionalContext = [
    "user_name": .string("Alice"),
    "turn_count": .int(3),
    "verbose": .bool(true),
]
// Override the sampler. Unset fields inherit from GenerationOptions.sampling.
mlxOptions.topP = 0.9
mlxOptions.topK = 40
mlxOptions.minP = 0.05
mlxOptions.repetitionPenalty = 1.1
mlxOptions.repetitionContextSize = 64
options[custom: MLXLanguageModel.self] = mlxOptions

let response = try await session.respond(
    to: "Summarize this transcript",
    options: options
)
```

You can specify `userInputProcessing` to enforce a consistent image
preprocessing step
(for example, fixed dimensions for predictable latency, memory usage, and vision behavior).
By default, images are passed through without an explicit resize override
(`resize: nil`), so MLX applies its default media processing behavior.

You can also set `additionalContext` to provide extra JSON template variables
for model-specific chat templates.

The sampling fields (`topP`, `topK`, `minP`, `repetitionPenalty`, `repetitionContextSize`)
override the matching values from `GenerationOptions.sampling`.
Leave them `nil` to inherit those values,
or to use MLX's defaults when neither is set.

GPU cache behavior can be configured when creating the model:

```swift
let model = MLXLanguageModel(
    modelId: "mlx-community/Qwen3.5-4B-MLX-4bit",
    gpuMemory: .automatic
)
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
    url: "https://github.com/huggingface/AnyLanguageModel.git",
    from: "1.2.0",
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
    url: "https://github.com/huggingface/AnyLanguageModel.git",
    from: "1.2.0",
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
    mirostat: .v2(tau: 5.0, eta: 0.1),  // Adaptive perplexity control
    assistantPrefill: "<think></think>"  // Text the response continues from
)

let response = try await session.respond(
    to: "Write a story",
    options: options
)
```

<a id="litert-lm-checkout-fails-with-a-git-lfs-smudge-error"></a>

### LiteRT-LM

LiteRT-LM support was added in 0.10.0 and removed in 0.11.0
because its dependency could prevent builds even when the backend was disabled.
See [upstream issue #2407](https://github.com/google-ai-edge/LiteRT-LM/issues/2407)
for details.
The `LiteRT` trait and `LiteRTLanguageModel` are no longer available.

### Ollama

Run models locally via Ollama's
[HTTP API](https://github.com/ollama/ollama/blob/main/docs/api.md):

```swift
// Default: connects to http://localhost:11434
let model = OllamaLanguageModel(model: "qwen3.5") // `ollama pull qwen3.5:9b`

// Custom endpoint
let model = OllamaLanguageModel(
    baseURL: URL(string: "http://remote-server:11434")!,
    model: "gemma4"
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
let model = OllamaLanguageModel(model: "qwen3.5") // `ollama pull qwen3.5:9b`
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
    model: "gpt-5.6-luna"
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
    model: "gpt-5.6-luna",
    apiVariant: .chatCompletions
)
```

Use custom generation options for advanced parameters like sampling controls,
reasoning effort (for GPT-5.6 and GPT-6 models), and vendor-specific extensions:

```swift
var options = GenerationOptions(temperature: 0.8)
options[custom: OpenAILanguageModel.self] = .init(
    topP: 0.9,
    frequencyPenalty: 0.5,
    presencePenalty: 0.3,
    stopSequences: ["END"],
    reasoningEffort: .high,        // For reasoning models (gpt-5.6, gpt-6-astra)
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
    model: "openai/gpt-5.6"
)

// Example: OpenAI
let model = OpenResponsesLanguageModel(
    baseURL: URL(string: "https://api.openai.com/v1/")!,
    apiKey: ProcessInfo.processInfo.environment["OPEN_RESPONSES_API_KEY"]!,
    model: "gpt-5.6-luna"
)

let session = LanguageModelSession(model: model)
let response = try await session.respond(to: "Say hello")
```

Custom options support Open Responses–specific fields,
such as `tool_choice` (including `allowed_tools`) and `extraBody`:

```swift
var options = GenerationOptions(temperature: 0.8)
options[custom: OpenResponsesLanguageModel.self] = .init(
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
    model: "claude-sonnet-5"
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
extended thinking, service tier, and effort:

```swift
var options = GenerationOptions(maximumResponseTokens: 8192)
options[custom: AnthropicLanguageModel.self] = .init(
    thinking: .enabled(budgetTokens: 4096),
    serviceTier: .priority
)
```

On models that support adaptive thinking, omit the token budget and use effort
to control how much work the model puts into its response:

```swift
var options = GenerationOptions(maximumResponseTokens: 8192)
options[custom: AnthropicLanguageModel.self] = .init(
    thinking: .adaptive(display: .omitted),
    effort: .medium
)
```

Effort levels are `.low`, `.medium`, `.high`, `.extraHigh`, and `.max`;
[support varies by model](https://platform.claude.com/docs/en/build-with-claude/effort).
The existing `Thinking(budgetTokens:)` initializer remains available.
Thinking blocks appear in responses and streaming snapshots
as [reasoning entries](#reasoning-in-the-transcript),
and signed blocks go back unchanged in tool-call follow-ups.

### Google Gemini

Uses the [Gemini API](https://ai.google.dev/api/generate-content) with Gemini models:

```swift
let model = GeminiLanguageModel(
    apiKey: ProcessInfo.processInfo.environment["GEMINI_API_KEY"]!,
    model: "gemini-3.8-flash"
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
> The `-skipMacroValidation` flag allows the `MLXHuggingFaceMacros` macro
> to run without prior approval in Xcode.
>
> Pass environment variables with `TEST_RUNNER_` prefix:
>
> ```bash
> export TEST_RUNNER_HF_TOKEN=your_huggingface_token
> xcodebuild test \
>   -skipMacroValidation \
>   -scheme AnyLanguageModel \
>   -destination 'platform=macOS' \
>   -only-testing:AnyLanguageModelTests/MLXLanguageModelTests
> ```

## Contributing

This is a community project and we welcome contributions.
Please check out
[Issues tagged with `good first issue`][good-first-issues]
if you are looking for a place to start!

Please ensure your code passes the build and test suite
before submitting a pull request.

[good-first-issues]: https://github.com/huggingface/AnyLanguageModel/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22

## License

[Apache 2](LICENSE).
