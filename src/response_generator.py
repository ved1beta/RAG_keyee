try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: google-generativeai package not found. Installing...")
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
    except Exception as e:
        print(f"Error installing google-generativeai: {str(e)}")
        GEMINI_AVAILABLE = False

import re

class ResponseGenerator:
    def __init__(self, api_key="AIzaSyCMJN_HqVPaHUEKeR_FfKxNwXhHcKXf-oE"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package is required but not available")
            
        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")  # Changed to gemini-pro as it's more widely available
        print("Initialized Gemini model successfully")

    def clean_response(self, response: str) -> str:
        """Clean the response"""
        try:
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
            prompt = f"""Based on the following context, provide a clear and concise answer to the question.

{context_text}

Question: {question}

Answer: """
            
            try:
                # Generate response using Gemini
                response = self.model.generate_content(prompt)
                
                if response.text:
                    return self.clean_response(response.text)
                else:
                    # If response is empty, try with a more detailed prompt
                    detailed_prompt = f"""Using the provided context, explain in detail:
                    
{context_text}

Question: {question}

Please cover:
1. Main concepts
2. Key points
3. Relevant details

Answer: """
                    
                    response = self.model.generate_content(detailed_prompt)
                    return self.clean_response(response.text)
                    
            except Exception as e:
                print(f"Error with Gemini API: {str(e)}")
                return f"Error generating response: {str(e)}"
            
        except Exception as e:
            print(f"Error in get_answer: {str(e)}")
            return f"Error: {str(e)}"

    def __del__(self):
        """
        Clean up resources when the object is deleted
        """
        if hasattr(self, 'model'):
            del self.model