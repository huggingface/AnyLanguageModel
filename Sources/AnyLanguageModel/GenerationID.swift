import struct Foundation.UUID

/// A unique identifier that is stable for the duration of a response, but not across responses.
///
/// The framework guarentees a `GenerationID` to be both present and stable when you
/// receive it from a `LanguageModelSession`. When you create an instance of
/// `GenerationID` there is no guarantee an identifier is present or stable.
///
/// ```swift
/// @Generable struct Person: Equatable {
///     var id: GenerationID
///     var name: String
/// }
///
/// struct PeopleView: View {
///     @State private var session = LanguageModelSession(model: SystemLanguageModel.default)
///     @State private var people = [Person.PartiallyGenerated]()
///
///     var body: some View {
///         // A person's name changes as the response is generated,
///         // and two people can have the same name, so it is not suitable
///         // for use as an id.
///         //
///         // `GenerationID` receives special treatment and is guaranteed
///         // to be both present and stable.
///         List {
///             ForEach(people) { person in
///                 Text("Name: \(person.name)")
///             }
///         }
///         .task {
///             for try! await people in stream.streamResponse(
///                 to: "Who were the first 3 presidents of the US?",
///                 generating: [Person].self
///             ) {
///                 withAnimation {
///                     self.people = people
///                 }
///             }
///         }
///     }
/// }
/// ```
///
/// - Note: The `Codable` conformance is exclusive to AnyLanguageModel
///   and using it means your code is no longer drop-in compatible
///   with the Foundation Models framework.
public struct GenerationID: Sendable, Hashable, Codable {
    private let uuid: UUID

    /// Create a new, unique `GenerationID`.
    public init() {
        self.uuid = UUID()
    }
}
