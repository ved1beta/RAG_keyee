import os
import uuid
import streamlit as st
from werkzeug.utils import secure_filename

# Import RAG components
from data_processing import PDFProcessor
from embeddings import EmbeddingGenerator
from query import QueryProcessor
from response_generator import ResponseGenerator

# Setup directories
current_dir = os.path.dirname(os.path.abspath(__file__))
upload_dir = os.path.join(current_dir, 'uploads')

# Create necessary directories
os.makedirs(upload_dir, exist_ok=True)

# Constants
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_rag_components(pdf_path):
    """Initialize RAG components for the uploaded PDF"""
    try:
        # Generate unique embeddings file name
        embeddings_file = os.path.join(os.path.dirname(pdf_path), f"embeddings_{uuid.uuid4()}.csv")
        
        # Initialize components
        pdf_processor = PDFProcessor(num_sentence_chunk_size=10)
        embedding_generator = EmbeddingGenerator()
        
        # Process PDF and generate embeddings
        st.info("Processing PDF...")
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
        
        st.success("PDF processed successfully!")
        return pdf_processor, embedding_generator, query_processor, response_generator
        
    except Exception as e:
        st.error(f"Error initializing RAG components: {str(e)}")
        raise

def main():
    st.title("PDF Question Answering System")
    
    # Initialize session state
    if 'rag_components' not in st.session_state:
        st.session_state.rag_components = None
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    
    if uploaded_file is not None:
        try:
            # Check file size
            file_size = len(uploaded_file.getvalue())
            if file_size > MAX_FILE_SIZE:
                st.error("File size exceeds the 16MB limit")
                return
            
            # Generate unique filename and save
            filename = secure_filename(f"{uuid.uuid4()}_{uploaded_file.name}")
            filepath = os.path.join(upload_dir, filename)
            
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize RAG components
            st.session_state.rag_components = initialize_rag_components(filepath)
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return
    
    # Query section
    if st.session_state.rag_components:
        query = st.text_input("Enter your question about the PDF:")
        
        if query:
            try:
                _, _, query_processor, response_generator = st.session_state.rag_components
                
                # Get relevant contexts
                contexts = query_processor.process_query(query, k=3)
                
                # Extract text from contexts
                context_texts = []
                for ctx in contexts:
                    if 'text' in ctx:
                        context_texts.append(ctx['text'])
                    elif 'sentence_chunk' in ctx:
                        context_texts.append(ctx['sentence_chunk'])
                
                # Generate response
                response = response_generator.get_answer(query, context_texts)
                
                # Display results
                st.write("### Answer:")
                st.write(response)
                
                # Display contexts in an expander
                with st.expander("View relevant contexts"):
                    for i, context in enumerate(contexts, 1):
                        st.write(f"Context {i}:")
                        st.write(context)
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    else:
        st.info("Please upload a PDF to start asking questions.")

if __name__ == '__main__':
    main()