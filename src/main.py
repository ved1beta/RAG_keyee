from response_generator import ResponseGenerator

def main():
    model = ResponseGenerator()
    
  
    question = ["What are macronutrients and their functions?"]
    
    print(f"\nQ: {question}")
    print(f"A: {model.get_answer(question)}")

if __name__ == "__main__":
    main()
    
