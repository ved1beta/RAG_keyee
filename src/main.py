import os
from data_processing import PDFProcessor
from embeddings import EmbeddingGenerator
from query import QueryProcessor
from retriever import SemanticRetriever
from response_generator import ResponseGenerator

def main():
    try:
        # Get the current directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the PDF file path
        pdf_path = os.path.join(current_dir, "data.pdf")
        
        # Check if PDF exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        print("1. Initializing components...")
        # Initialize components
        pdf_processor = PDFProcessor(num_sentence_chunk_size=10)
        embedding_generator = EmbeddingGenerator()
        query_processor = QueryProcessor(
            embeddings_path="text_chunks_and_embeddings_df.csv",
            similarity_metric='cosine',
            min_score_threshold=0.1
        )
        response_generator = ResponseGenerator()
        
        print("\n2. Processing PDF and generating embeddings...")
        # Process PDF and generate embeddings if needed
        if not os.path.exists("text_chunks_and_embeddings_df.csv"):
            print("Processing PDF and generating new embeddings...")
            df = pdf_processor.process_pdf(pdf_path)
            embeddings_df = embedding_generator.generate_embeddings(df)
            embedding_generator.save_embeddings(
                embeddings_df, 
                "text_chunks_and_embeddings_df.csv"
            )
        else:
            print("Using existing embeddings file...")

        print("\n3. Starting Q&A Session...")
        # Example queries
        queries = [
            "what is sklearn and how to use it ? "
        ]

        for query in queries:
            print("\n" + "="*50)
            print(f"Query: {query}")
            print("="*50)
            
            # Retrieve relevant contexts
            retrieved_contexts = query_processor.process_query(query, k=3)
            
            print("\nRetrieved Contexts:")
            for i, context in enumerate(retrieved_contexts, 1):
                print(f"\nContext {i}:")
                print(f"Similarity Score: {context['similarity_score']:.4f}")
                if context.get('page_number') is not None:
                    print(f"Page Number: {context['page_number']}")
                print(f"Text: {context['text'][:200]}...")  # Truncate long texts
            
            # Generate response using the LLM
            print("\nGenerating response...")
            response = response_generator.get_answer(query)
            
            print("\nGenerated Response:")
            print("-"*30)
            print(response)
            print("-"*30)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    
