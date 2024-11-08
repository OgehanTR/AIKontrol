import customtkinter as ctk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class CodeAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Kontrol")
        self.geometry("600x400")
        self.configure(bg="black")
        self.attributes("-alpha", 0.9)

        self.code_input = ctk.CTkTextbox(self, width=550, height=200, font=("Courier", 12))
        self.code_input.insert("1.0", "Kodu Buraya Yapıştırınız")
        self.code_input.pack(pady=20)

        self.analyze_button = ctk.CTkButton(self, text="Kodu Analiz Et", command=self.analyze_code)
        self.analyze_button.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def analyze_code(self):
        code_text = self.code_input.get("1.0", "end").strip()
        if not code_text:
            self.result_label.configure(text="Lütfen bir kod girin!")
            return

        is_ai_generated = self.detect_ai_code(code_text)
        
        if is_ai_generated:
            self.result_label.configure(text="Bu kod,AI tarafından yazılmış.", text_color="red")
        else:
            self.result_label.configure(text="Bu kod,insan tarafından yazılmış.", text_color="green")
    
    def detect_ai_code(self, code_text):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-base")
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/CodeBERT-base")
        
        inputs = tokenizer(code_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        
        return prediction == 1

if __name__ == "__main__":
    app = CodeAnalyzerApp()
    app.mainloop()
