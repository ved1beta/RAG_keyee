import os
import uuid
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename


from data_processing import PDFProcessor
from embeddings import EmbeddingGenerator
from query import QueryProcessor
from response_generator import ResponseGenerator


current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
upload_dir = os.path.join(current_dir, 'uploads')

app = Flask(__name__, template_folder=template_dir)
CORS(app)

ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = upload_dir
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


os.makedirs(template_dir, exist_ok=True)
os.makedirs(upload_dir, exist_ok=True)


current_pdf_path = None
pdf_processor = None
embedding_generator = None
query_processor = None
response_generator = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_rag_components(pdf_path):
    global pdf_processor, embedding_generator, query_processor, response_generator, current_pdf_path
    try:
        embeddings_file = os.path.join(os.path.dirname(pdf_path), f"embeddings_{uuid.uuid4()}.csv")
        
       
        pdf_processor = PDFProcessor(num_sentence_chunk_size=10)
        embedding_generator = EmbeddingGenerator()
        
        
        print(f"Processing PDF: {pdf_path}")
        df = pdf_processor.process_pdf(pdf_path)
        print("Generating embeddings...")
        embeddings_df = embedding_generator.generate_embeddings(df)
        embedding_generator.save_embeddings(embeddings_df, embeddings_file)
        
        
        query_processor = QueryProcessor(
            embeddings_path=embeddings_file,
            similarity_metric='cosine',
            min_score_threshold=0.1
        )
        
     
        response_generator = ResponseGenerator()
        
     
        current_pdf_path = pdf_path
        print("RAG components initialized successfully")
        
    except Exception as e:
        print(f"Error initializing RAG components: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            try:
               
                filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
               
                print(f"Saving file to: {filepath}")
                file.save(filepath)
                
               
                print("Initializing RAG components...")
                initialize_rag_components(filepath)
                
                return jsonify({
                    "message": "PDF uploaded and processed successfully",
                    "filename": filename
                }), 200
                
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500
        
        return jsonify({"error": "File type not allowed"}), 400
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({"error": f"Upload error: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def process_query():
    try:
        if not query_processor or not response_generator:
            return jsonify({"error": "Please upload a PDF first"}), 400
        
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data['query']
        print(f"Processing query: {query}")
        
  
        contexts = query_processor.process_query(query, k=3)
        
        context_texts = []
        for ctx in contexts:
            if 'text' in ctx:
                context_texts.append(ctx['text'])
            elif 'sentence_chunk' in ctx:
                context_texts.append(ctx['sentence_chunk'])

        response = response_generator.get_answer(query, context_texts)
        
        return jsonify({
            "query": query,
            "response": response,
            "contexts": contexts
        }), 200
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500

if __name__ == '__main__':
    print(f"Template folder: {template_dir}")
    print(f"Upload folder: {upload_dir}")
    
    # Verify template exists
    template_path = os.path.join(template_dir, 'index.html')
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found at: {template_path}")
        
    app.run(debug=True, port=5000)
