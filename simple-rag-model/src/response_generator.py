import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import importlib.util

class ResponseGenerator:
    def __init__(self, model_id="google/gemma-2b-it"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token="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"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token="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",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def get_answer(self, question: str) -> str:
        """
        Get direct answer for a question
        """
        try:
            # Format the prompt
            prompt = f"<start_of_turn>user\n{question}\n<end_of_turn>\n<start_of_turn>model\n"
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Get response text
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            response = response.replace("<end_of_turn>", "").strip()
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"