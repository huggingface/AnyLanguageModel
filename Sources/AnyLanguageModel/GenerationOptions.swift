import JSONSchema

/// Options that control how the model generates its response to a prompt.
///
/// Create a ``GenerationOptions`` structure when you want to adjust
/// the way the model generates its response. Use this structure to
/// perform various adjustments on how the model chooses output tokens,
/// to specify the penalties for repeating tokens or generating
/// longer responses.
public struct GenerationOptions: Sendable, Equatable {
    /// A type that defines how values are sampled from a probability distribution.
    ///
    /// A model builds its response to a prompt in a loop. At each iteration in the
    /// loop the model produces a probability distribution for all the tokens in its
    /// vocabulary. The sampling mode controls how a token is selected from that
    /// distribution.
    public struct SamplingMode: Sendable, Equatable {
        enum Mode: Equatable, Codable {
            case greedy
            case topK(Int, seed: UInt64?)
            case nucleus(Double, seed: UInt64?)
        }

        let mode: Mode

        /// A sampling mode that always chooses the most likely token.
        ///
        /// Using this mode will always result in the same output
        /// for a given input. Responses produced with greedy sampling
        /// are statistically likely, but may lack the human-like quality
        /// and variety of other sampling strategies.
        public static var greedy: SamplingMode {
            SamplingMode(mode: .greedy)
        }

        /// A sampling mode that considers a fixed number of high-probability tokens.
        ///
        /// Also known as top-k.
        ///
        /// During the token-selection process, the vocabulary is sorted by probability a
        /// token is selected from among the top K candidates. Smaller values of K will
        /// ensure only the most probable tokens are candidates for selection, resulting
        /// in more deterministic and confident answers. Larger values of K will allow less
        /// probably tokens to be selected, raising non-determinism and creativity.
        ///
        /// - Note: Setting a random seed is not guaranteed to result in fully deterministic
        ///   output. It is best effort.
        ///
        /// - Parameters:
        ///   - top: The number of tokens to consider.
        ///   - seed: An optional random seed used to make output more deterministic.
        public static func random(top k: Int, seed: UInt64? = nil) -> SamplingMode {
            SamplingMode(mode: .topK(k, seed: seed))
        }

        /// A mode that considers a variable number of high-probability tokens
        /// based on the specified threshold.
        ///
        /// Also known as top-p or nucleus sampling.
        ///
        /// With nucleus sampling, tokens are sorted by probability and added to a
        /// pool of candidates until the cumulative probability of the pool exceeds
        /// the specified threshold, and then a token is sampled from the pool.
        ///
        /// Because the number of tokens isn't predetermined, the selection pool size
        /// will be larger when the distribution is flat and smaller when it is spikey.
        /// This variability can lead to a wider variety of options to choose from, and
        /// potentially more creative responses.
        ///
        /// - Note: Setting a random seed is not guaranteed to result in fully deterministic
        ///   output. It is best effort.
        ///
        /// - Parameters:
        ///     - probabilityThreshold: A number between `0.0` and `1.0` that
        ///       increases sampling pool size.
        ///     - seed: An optional random seed used to make output more deterministic.
        public static func random(probabilityThreshold: Double, seed: UInt64? = nil) -> SamplingMode {
            SamplingMode(mode: .nucleus(probabilityThreshold, seed: seed))
        }
    }

    /// A sampling strategy for how the model picks tokens when generating a
    /// response.
    ///
    /// When you execute a prompt on a model, the model produces a probability
    /// for every token in its vocabulary. The sampling strategy controls how
    /// the model narrows down the list of tokens to consider during that process.
    /// A strategy that picks the single most likely token yields a predictable
    /// response every time, but other strategies offer results that often
    /// sound more natural to a person.
    ///
    /// - Note: Leaving the `sampling` nil lets the system choose a
    ///   a reasonable default on your behalf.
    public var sampling: SamplingMode?

    /// Temperature influences the confidence of the models response.
    ///
    /// The value of this property must be a number between `0` and `1` inclusive.
    ///
    /// Temperature is an adjustment applied to the probability distribution
    /// prior to sampling. A value of `1` results in no adjustment. Values less
    /// than `1` will make the probability distribution sharper, with already
    /// likely tokens becoming even more likely.
    ///
    /// The net effect is that low temperatures manifest as more stable and
    /// predictable responses, while high temperatures give the model more
    /// creative license.
    ///
    /// - Note: Leaving `temperature` nil lets the system choose a reasonable
    ///   default on your behalf.
    public var temperature: Double?

    /// The maximum number of tokens the model is allowed to produce in its response.
    ///
    /// If the model produce `maximumResponseTokens` before it naturally completes its response,
    /// the response will be terminated early. No error will be thrown. This property
    /// can be used to protect against unexpectedly verbose responses and runaway generations.
    ///
    /// If no value is specified, then the model is allowed to produce the longest answer
    /// its context size supports. If the response exceeds that limit without terminating,
    /// an error will be thrown.
    public var maximumResponseTokens: Int?

    /// Storage for model-specific custom options.
    private var customOptionsStorage: CustomOptionsStorage = .init()

    /// Accesses model-specific custom generation options.
    ///
    /// Use this subscript to set or retrieve custom options that are specific
    /// to a particular language model implementation.
    ///
    /// ```swift
    /// var options = GenerationOptions(temperature: 0.7)
    /// options[custom: OpenAILanguageModel.self] = .init(
    ///     extraBody: ["reasoning": ["enabled": true]]
    /// )
    /// ```
    ///
    /// - Parameter modelType: The language model type to get or set custom options for.
    /// - Returns: The custom options for the specified model type, or `nil` if none are set.
    public subscript<Model: LanguageModel>(
        custom modelType: Model.Type
    ) -> Model.CustomGenerationOptions? {
        get {
            customOptionsStorage[Model.self]
        }
        set {
            customOptionsStorage[Model.self] = newValue
        }
    }

    /// Creates generation options that control token sampling behavior.
    ///
    /// - Parameters:
    ///   - sampling: A strategy to use for sampling from a distribution.
    ///   - temperature: Increasing temperature makes it possible for the model to produce less likely
    ///     responses. Must be between `0` and `1`, inclusive.
    ///   - maximumResponseTokens: The maximum number of tokens the model is allowed
    ///     to produce before being artificially halted. Must be positive.
    public init(
        sampling: SamplingMode? = nil,
        temperature: Double? = nil,
        maximumResponseTokens: Int? = nil
    ) {
        self.sampling = sampling
        self.temperature = temperature
        self.maximumResponseTokens = maximumResponseTokens
    }
}

// MARK: - Transcript Coding

extension GenerationOptions {
    /// The coded form of generation options in a transcript prompt.
    ///
    /// `GenerationOptions` isn't `Codable`, matching Foundation Models,
    /// but ``Transcript/Prompt`` is, so it codes its options through this type.
    /// It codes the sampling mode, temperature, and maximum response tokens
    /// in the same format as earlier releases.
    /// Custom options aren't coded,
    /// because decoding them would need a registry of every model's option types.
    struct TranscriptCoding: Codable {
        struct Sampling: Codable {
            var mode: SamplingMode.Mode
        }

        var sampling: Sampling?
        var temperature: Double?
        var maximumResponseTokens: Int?

        init(_ options: GenerationOptions) {
            self.sampling = options.sampling.map { Sampling(mode: $0.mode) }
            self.temperature = options.temperature
            self.maximumResponseTokens = options.maximumResponseTokens
        }

        var options: GenerationOptions {
            GenerationOptions(
                sampling: sampling.map { SamplingMode(mode: $0.mode) },
                temperature: temperature,
                maximumResponseTokens: maximumResponseTokens
            )
        }
    }
}

// MARK: - Custom Generation Options

/// A protocol for model-specific generation options.
///
/// Conform to this protocol to define custom options that are specific to a
/// particular language model implementation. These options can be accessed
/// through the ``GenerationOptions`` subscript.
///
/// ```swift
/// extension MyLanguageModel {
///     public struct CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions {
///         public var customParameter: String?
///     }
/// }
/// ```
public protocol CustomGenerationOptions: Equatable, Sendable {}

extension Never: CustomGenerationOptions {}

extension Dictionary: CustomGenerationOptions where Key == String, Value == JSONValue {}

/// Storage for model-specific custom options.
private struct CustomOptionsStorage: Sendable, Equatable {
    private var storage: [ObjectIdentifier: AnyCustomOptions] = [:]

    init() {}

    subscript<Model: LanguageModel>(modelType: Model.Type) -> Model.CustomGenerationOptions? {
        get {
            guard let wrapper = storage[ObjectIdentifier(modelType)] else {
                return nil
            }
            return wrapper.value as? Model.CustomGenerationOptions
        }
        set {
            if let newValue {
                storage[ObjectIdentifier(modelType)] = AnyCustomOptions(newValue)
            } else {
                storage.removeValue(forKey: ObjectIdentifier(modelType))
            }
        }
    }

    static func == (lhs: CustomOptionsStorage, rhs: CustomOptionsStorage) -> Bool {
        guard lhs.storage.count == rhs.storage.count else { return false }
        for (key, lhsWrapper) in lhs.storage {
            guard let rhsWrapper = rhs.storage[key],
                lhsWrapper.isEqual(to: rhsWrapper)
            else {
                return false
            }
        }
        return true
    }
}

// MARK: - AnyCustomOptions

/// A type-erased wrapper for custom generation options.
private struct AnyCustomOptions: Sendable {
    let value: any CustomGenerationOptions
    let equalsImpl: @Sendable (any CustomGenerationOptions) -> Bool

    init<T: CustomGenerationOptions>(_ value: T) {
        self.value = value
        self.equalsImpl = { other in
            guard let otherTyped = other as? T else { return false }
            return value == otherTyped
        }
    }

    func isEqual(to other: AnyCustomOptions) -> Bool {
        equalsImpl(other.value)
    }
}
