import Foundation
import Testing

@testable import AnyLanguageModel

#if Llama
    @Suite("LlamaLanguageModel Gemma 4 template")
    struct LlamaGemma4TemplateTests {
        @Test func rendersTurnsAndOpensTheModelTurn() {
            let rendered = LlamaLanguageModel.renderGemma4Prompt(
                messages: [
                    (role: "system", content: "Be brief."),
                    (role: "user", content: "Hi"),
                    (role: "assistant", content: "Hello"),
                    (role: "user", content: "Bye"),
                ],
                assistantPrefill: nil
            )
            #expect(
                rendered
                    == "<|turn>system\nBe brief.<turn|>\n"
                    + "<|turn>user\nHi<turn|>\n"
                    + "<|turn>model\nHello<turn|>\n"
                    + "<|turn>user\nBye<turn|>\n"
                    + "<|turn>model\n"
            )
        }

        @Test func appendsTheAssistantPrefill() {
            let rendered = LlamaLanguageModel.renderGemma4Prompt(
                messages: [(role: "user", content: "Hi")],
                assistantPrefill: "<think></think>"
            )
            #expect(rendered == "<|turn>user\nHi<turn|>\n<|turn>model\n<think></think>")
        }
    }
#endif
