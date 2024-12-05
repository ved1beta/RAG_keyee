#import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM
#import importlib.util
#
#class ResponseGenerator:
#    def __init__(self, model_id="google/gemma-2b-it"):
#        self.device = "cuda" if torch.cuda.is_available() else "cpu"
#        
#        # Load tokenizer and model
#        self.tokenizer = AutoTokenizer.from_pretrained(
#            model_id,
#            token="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"
#        )
#        self.model = AutoModelForCausalLM.from_pretrained(
#            model_id,
#            token="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",
#            torch_dtype=torch.float16,
#            device_map="auto"
#        )
#
#    def get_answer(self, question: str) -> str:
#        """
#        Get direct answer for a question
#        """
#        try:
#            # Format the prompt
#            prompt = f"<start_of_turn>user\n{question}\n<end_of_turn>\n<start_of_turn>model\n"
#            
#            # Generate response
#            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#            outputs = self.model.generate(
#                **inputs,
#                max_new_tokens=512,
#                temperature=0.7,
#                do_sample=True,
#                pad_token_id=self.tokenizer.eos_token_id
#            )
#            
#            # Get response text
#            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#            response = response.replace(prompt, "").strip()
#            response = response.replace("<end_of_turn>", "").strip()
#            
#            return response
#            
#        except Exception as e:
#            return f"Error: {str(e)}"

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import numpy as np

class ResponseGenerator:
    def __init__(self, model_name='distilbert-base-uncased-distilled-squad'):
        """
        Initialize local model for question answering
        
        :param model_name: Hugging Face model name
        """
        print(f"Loading model: {model_name}")
        
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
            # Move model to device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_answer(self, query: str, contexts: list = None) -> str:
        """
        Generate response using local model
        
        :param query: User's query
        :param contexts: List of context passages
        :return: Generated response
        """
        if not contexts:
            return "Please provide some context to answer the question."
        
        try:
            # Select the most relevant context (first one)
            context = contexts[0]
            
            # Tokenize inputs
            inputs = self.tokenizer.encode_plus(
                query, 
                context, 
                add_special_tokens=True, 
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process answer
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits)
            
            # Decode the answer
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    inputs['input_ids'][0][answer_start:answer_end+1]
                )
            )
            
            # If no clear answer found, provide a generic response
            if not answer.strip():
                return "I couldn't find a specific answer in the given context. Could you rephrase your question?"
            
            return answer
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"An error occurred while generating the response: {str(e)}"

    def __del__(self):
        """
        Clean up resources when the object is deleted
        """
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer