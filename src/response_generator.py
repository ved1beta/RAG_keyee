import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

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

    def clean_response(self, response: str, prompt: str) -> str:
        """Clean the response by removing prompt and special tokens"""
        try:
            # Remove everything before the actual response
            if "Based on the provided context, here's the explanation:" in response:
                response = response.split("Based on the provided context, here's the explanation:", 1)[1]
            elif "Here's a comprehensive explanation:" in response:
                response = response.split("Here's a comprehensive explanation:", 1)[1]
            
            # Remove special tokens and clean up
            response = response.replace("<end_of_turn>", "")
            response = response.replace("<start_of_turn>user", "")
            response = response.replace("<start_of_turn>model", "")
            
            # Remove the original prompt and question
            response = response.replace(prompt, "")
            
            # Remove context prefixes and content
            response = re.sub(r'Context \d+:.*?\n', '', response, flags=re.DOTALL)
            
            # Remove the question from the response
            response = re.sub(r'Question:.*?\n', '', response, flags=re.DOTALL)
            
            # Clean up extra whitespace and newlines
            response = re.sub(r'\s+', ' ', response)
            response = response.strip()
            
            return response
            
        except Exception as e:
            print(f"Error cleaning response: {str(e)}")
            return response.strip()

    def get_answer(self, question: str, contexts: list = None) -> str:
        """Get answer for a question using provided contexts"""
        try:
            if not contexts:
                return "Please provide some context to answer the question."
            
            # Format contexts
            formatted_contexts = []
            for i, ctx in enumerate(contexts, 1):
                clean_ctx = ctx.strip().replace("\n", " ")
                clean_ctx = re.sub(r'\s+', ' ', clean_ctx)
                formatted_contexts.append(f"Context {i}: {clean_ctx}")
            
            context_text = "\n\n".join(formatted_contexts)
            
            # Create prompt
            prompt = f"""<start_of_turn>user
Read the following context and answer the question concisely.

{context_text}

Question: {question}
<end_of_turn>
<start_of_turn>model
Based on the provided context, here's the explanation:"""
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                top_p=0.9,
                top_k=50
            )
            
            # Get and clean response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            clean_response = self.clean_response(full_response, prompt)
            
            # If response is too short, try with more detailed prompt
            if len(clean_response.split()) < 20:
                detailed_prompt = f"""<start_of_turn>user
Please provide a detailed explanation of {question} based on this context:

{context_text}

Focus on:
1. What is it?
2. How is it structured?
3. Key components
4. How does it work?
<end_of_turn>
<start_of_turn>model
Here's a comprehensive explanation:"""
                
                inputs = self.tokenizer(detailed_prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    top_p=0.9,
                    top_k=50
                )
                
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                clean_response = self.clean_response(full_response, detailed_prompt)
            
            return clean_response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

    def __del__(self):
        """
        Clean up resources when the object is deleted
        """
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer