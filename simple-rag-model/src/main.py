import os
from data_processing import PDFProcessor
from embeddings import EmbeddingGenerator
from query_processing import QueryProcessor

def main():
    # Get the current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the PDF file path
    pdf_path = os.path.join(current_dir, "data.pdf")
    
    # Initialize PDF Processor
    pdf_processor = PDFProcessor(num_sentence_chunk_size=10)
    
    # Process PDF and get DataFrame
    df = pdf_processor.process_pdf(pdf_path)
    
    
    embedding_generator = EmbeddingGenerator()
    embeddings_df = embedding_generator.generate_embeddings(df)
    embedding_generator.save_embeddings(
        embeddings_df, 
        "text_chunks_and_embeddings_df.csv"
    )

    query_processor = QueryProcessor(
        embeddings_path="text_chunks_and_embeddings_df.csv"
    )

    query = "macronutrients functions"
    retrieved_contexts = query_processor.process_query(query, k=5)

    print(f"\nQuery: {query}")
    print("\nTop Retrieved Contexts:")
    for i, context in enumerate(retrieved_contexts, 1):
        print(f"\nContext {i} (Similarity: {context['similarity_score']:.4f}):")
        print(context['text'])

if __name__ == "__main__":
    main()