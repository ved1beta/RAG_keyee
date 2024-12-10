import os
import streamlit as st
import uuid

# Import your existing RAG components
from data_processing import PDFProcessor
from embeddings import EmbeddingGenerator
from query import QueryProcessor
from response_generator import ResponseGenerator

def initialize_rag_components(pdf_path):
    """Initialize RAG components for the uploaded PDF"""
    try:
        # Generate unique embeddings file name
        embeddings_file = os.path.join(os.path.dirname(pdf_path), f"embeddings_{uuid.uuid4()}.csv")
        
        # Initialize components
        pdf_processor = PDFProcessor(num_sentence_chunk_size=10)
        embedding_generator = EmbeddingGenerator()
        
        # Process PDF and generate embeddings
        st.info(f"Processing PDF: {pdf_path}")
        df = pdf_processor.process_pdf(pdf_path)
        st.info("Generating embeddings...")
        embeddings_df = embedding_generator.generate_embeddings(df)
        embedding_generator.save_embeddings(embeddings_df, embeddings_file)
        
        # Initialize query processor with new embeddings
        query_processor = QueryProcessor(
            embeddings_path=embeddings_file,
            similarity_metric='cosine',
            min_score_threshold=0.1
        )
        
        # Initialize response generator
        response_generator = ResponseGenerator()
        
        return pdf_processor, embedding_generator, query_processor, response_generator
    
    except Exception as e:
        st.error(f"Error initializing RAG components: {str(e)}")
        return None, None, None, None

def main():
    st.title("PDF Question Answering Assistant")

    # Sidebar for PDF upload
    st.sidebar.header("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    # Initialize session state variables
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = None
        st.session_state.embedding_generator = None
        st.session_state.query_processor = None
        st.session_state.response_generator = None

    # PDF Upload and Processing
    if uploaded_file is not None:
        # Create a temporary file
        temp_pdf_path = os.path.join("uploads", f"{uuid.uuid4()}_{uploaded_file.name}")
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded file
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize RAG components
        pdf_processor, embedding_generator, query_processor, response_generator = initialize_rag_components(temp_pdf_path)
        
        # Store in session state
        st.session_state.pdf_processor = pdf_processor
        st.session_state.embedding_generator = embedding_generator
        st.session_state.query_processor = query_processor
        st.session_state.response_generator = response_generator
        
        st.sidebar.success("PDF processed successfully!")

    # Query Section
    st.header("Ask a Question")
    query = st.text_input("Enter your question about the PDF")

    if query and st.session_state.query_processor and st.session_state.response_generator:
        with st.spinner("Generating answer..."):
            # Get relevant contexts
            contexts = st.session_state.query_processor.process_query(query, k=3)
            
            # Extract text from contexts
            context_texts = [ctx.get('text', ctx.get('sentence_chunk', '')) for ctx in contexts]
            
            # Generate response
            response = st.session_state.response_generator.get_answer(query, context_texts)
            
            # Display response
            st.subheader("Answer")
            st.write(response)
            
            # Optional: Show context details
            with st.expander("Context Details"):
                for i, context in enumerate(contexts, 1):
                    st.text(f"Context {i}: {context.get('text', context.get('sentence_chunk', 'No text available'))}")

if __name__ == "__main__":
    main()