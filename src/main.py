from response_generator import ResponseGenerator

def main():
    # Initialize the model
    model = ResponseGenerator()
    
    # Example questions
    questions = [
        "What are macronutrients and their functions?",
        "Explain the role of proteins in the body",
        "How do carbohydrates provide energy?"
    ]
    
    # Get answers
    for question in questions:
        print(f"\nQ: {question}")
        print(f"A: {model.get_answer(question)}")

if __name__ == "__main__":
    main()
    
