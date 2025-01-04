import os
from groq import Groq
import re

class ResponseGenerator:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = "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"
        
        if not api_key:
            raise ValueError("Groq API key is required")
            
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"  # Default model
        print("Initialized Groq client successfully")

    def clean_response(self, response: str) -> str:
        try:
            response = re.sub(r'\s+', ' ', response)
            response = response.strip()
            return response
            
        except Exception as e:
            print(f"Error cleaning response: {str(e)}")
            return response.strip()

    def get_answer(self, question: str, contexts: list = None) -> str:
        """Get answer for a question using provided contexts"""
        try:
            print("Debug - Received contexts:", contexts)
            print("Debug - Context type:", type(contexts))
            if contexts:
                print("Debug - First context type:", type(contexts[0]))
                
            if not contexts:
                return "Please provide some context to answer the question."
            
            formatted_contexts = []
            for i, ctx in enumerate(contexts, 1):
                clean_ctx = ctx.strip().replace("\n", " ")
                clean_ctx = re.sub(r'\s+', ' ', clean_ctx)
                formatted_contexts.append(f"Context {i}: {clean_ctx}")
            
            context_text = "\n\n".join(formatted_contexts)
            
            prompt = f"""Based on the following context, provide a clear and concise answer to the question.

{context_text}

Question: {question}

Answer: """
            
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model
                )
                
                response = chat_completion.choices[0].message.content
                
                if response:
                    return self.clean_response(response)
                else:
                    detailed_prompt = f"""Using the provided context, explain in detail:
                    
{context_text}

Question: {question}

Please cover:
1. Main concepts
2. Key points
3. Relevant details

Answer: """
                    
                    chat_completion = self.client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": detailed_prompt
                            }
                        ],
                        model=self.model
                    )
                    response = chat_completion.choices[0].message.content
                    return self.clean_response(response)
                    
            except Exception as e:
                print(f"Error with Groq API: {str(e)}")
                return f"Error generating response: {str(e)}"
            
        except Exception as e:
            print(f"Error in get_answer: {str(e)}")
            return f"Error: {str(e)}"

    def __del__(self):
        """
        Clean up resources when the object is deleted
        """
        if hasattr(self, 'client'):
            del self.client