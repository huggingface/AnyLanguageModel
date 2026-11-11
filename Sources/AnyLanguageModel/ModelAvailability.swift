import Foundation
import HTTPTypes

/// Whether a model can be used now, as asking where it runs says; when it can't, why.
///
/// A model's ``LanguageModel/availability`` says what it knows without asking anyone, so a
/// model that runs on a provider's servers always says it's available. Its
/// ``LanguageModel/checkAvailability()`` asks: the provider, whether it takes the key and has
/// the model; the device, whether it can run it.
///
/// - Note: This API is exclusive to AnyLanguageModel
///   and using it means your code is no longer drop-in compatible
///   with the Foundation Models framework.
public enum ModelAvailability: Sendable {
    case available
    case unavailable(ModelUnavailability)
}

/// Why a model can't be used now.
///
/// - Note: This API is exclusive to AnyLanguageModel
///   and using it means your code is no longer drop-in compatible
///   with the Foundation Models framework.
public struct ModelUnavailability: Error, Sendable, CustomStringConvertible {
    /// What's in the way. More kinds can be added, so a `switch` over them needs a `default`.
    public struct Kind: Hashable, Sendable, CustomStringConvertible {
        public let description: String

        private init(_ description: String) {
            self.description = description
        }

        /// The device can't run the model.
        public static let deviceNotEligible = Kind("Device not eligible")
        /// Apple Intelligence is turned off.
        public static let appleIntelligenceNotEnabled = Kind("Apple Intelligence not enabled")
        /// The model or the system isn't ready yet, such as while the model downloads.
        public static let notReady = Kind("Not ready")
        /// The model was loaded before, and loading it failed.
        public static let failedToLoad = Kind("Failed to load")
        /// The model named isn't where it's asked for: a provider that doesn't have it, or a file
        /// that isn't there.
        public static let modelNotFound = Kind("Model not found")
        /// The provider asks for a key, and none was given.
        public static let keyNeeded = Kind("Key needed")
        /// The provider refused the key given.
        public static let keyRefused = Kind("Key refused")
        /// The provider couldn't be reached.
        public static let unreachable = Kind("Unreachable")
        /// The provider answered with something else: the underlying error has its answer.
        public static let providerError = Kind("Provider error")
        /// A reason only the model knows: its ``LanguageModel/availability`` says which.
        public static let other = Kind("Other")
    }

    /// What's in the way.
    public let kind: Kind

    /// The error that says more, when there's one: the provider's answer, or the transport's.
    public let underlyingError: (any Error)?

    public init(kind: Kind, underlyingError: (any Error)? = nil) {
        self.kind = kind
        self.underlyingError = underlyingError
    }

    public var description: String {
        guard let underlyingError else { return kind.description }
        return "\(kind.description): \(underlyingError)"
    }
}

extension ModelUnavailability {
    /// What `error`, from asking a provider about a model, says: that the key was missing or
    /// refused, that the provider doesn't have the model, that it couldn't be reached, or what it
    /// answered instead. A request that was cancelled throws.
    ///
    /// - Parameters:
    ///   - keyGiven: Whether the request carried a key: a refusal without one asks for a key.
    ///   - missingModel: Whether a `404 Not Found` means the provider doesn't have the model,
    ///     as it does when the model itself was asked for.
    init(asking error: any Error, keyGiven: Bool, missingModel: Bool) throws(CancellationError) {
        if error is CancellationError || Task.isCancelled { throw CancellationError() }
        guard let error = error as? HTTPTransportError else {
            // It failed on the way, before the provider could answer.
            self.init(kind: .unreachable, underlyingError: error)
            return
        }
        switch error.response?.status {
        case .unauthorized?, .forbidden?:
            self.init(kind: keyGiven ? .keyRefused : .keyNeeded, underlyingError: error)
        case .notFound? where missingModel:
            self.init(kind: .modelNotFound, underlyingError: error)
        default:
            self.init(kind: .providerError, underlyingError: error)
        }
    }
}
