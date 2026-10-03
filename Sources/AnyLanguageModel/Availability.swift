/// The availability status for a specific language model.
///
/// This generic type replaces Foundation Models' `SystemLanguageModel.Availability`
/// so that each model can report its own reasons for being unavailable.
/// The cases have the same names,
/// so a `switch` over a model's `availability` works with both frameworks.
///
/// - Note: This API is exclusive to AnyLanguageModel
///   and using it means your code is no longer drop-in compatible
///   with the Foundation Models framework.
public enum Availability<UnavailableReason> {
    /// The model is ready for making requests.
    case available

    /// Indicates that the model is not ready for requests.
    case unavailable(UnavailableReason)
}

extension Availability: Equatable where UnavailableReason: Equatable {}
extension Availability: Hashable where UnavailableReason: Hashable {}
extension Availability: Sendable where UnavailableReason: Sendable {}
