# RAG Model for Research Paper Summarizing

A Retrieval-Augmented Generation (RAG) system that helps users understand and analyze research papers. Upload a PDF of a research paper and ask questions about its content.

![RAG Model Demo](demo.gif)

## Features

- 📄 PDF Research Paper Processing
- 🔍 Semantic Search with Embeddings
- 💡 Intelligent Question Answering
- 🤖 Gemini Pro Integration
- 🎯 Context-Aware Responses
- 🌐 Web Interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/simple-rag-model.git
cd simple-rag-model
```

2. Create and activate a virtual environment:
```bash
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

Create a `requirements.txt` file with:
```txt
flask==2.0.1
flask-cors==3.0.10
google-generativeai==0.3.0
numpy==1.24.3
pandas==2.0.2
PyMuPDF==1.22.5
python-dotenv==1.0.0
sentence-transformers==2.2.2
torch==2.0.1
tqdm==4.65.0
```

## Environment Setup

1. Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

2. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

1. Start the server:
```bash
python src/application.py
```

2. Open your browser and go to:
```
http://localhost:5000
```

3. Upload a research paper PDF and start asking questions!

## Project Structure

```
simple-rag-model/
├── src/
│   ├── templates/
│   │   └── index.html
│   ├── application.py
│   ├── data_processing.py
│   ├── embeddings.py
│   ├── query.py
│   ├── response_generator.py
│   └── retriever.py
├── uploads/
├── requirements.txt
└── README.md
```

## How It Works

1. **PDF Processing**: 
   - Extracts text from PDF
   - Splits into meaningful chunks
   - Maintains document structure

2. **Embedding Generation**:
   - Uses SentenceTransformers
   - Creates semantic embeddings
   - Enables similarity search

3. **Query Processing**:
   - Processes user questions
   - Finds relevant contexts
   - Ranks by similarity

4. **Response Generation**:
   - Uses Gemini Pro API
   - Context-aware responses
   - Natural language answers

## API Endpoints

- `POST /upload`: Upload PDF file
- `POST /query`: Process questions about the PDF

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini API for text generation
- Sentence Transformers for embeddings
- Flask for web framework
- PyMuPDF for PDF processing

## Contact

Your Name - [@ant_vedaya](https://twitter.com/ant_vedaya)
GitHub: [@ved1beta](https://github.com/ved1beta)

Project Link: [https://github.com/ved1beta/simple-rag-model](https://github.com/ved1beta/simple-rag-model)
```

To create a demo video:
1. Record your screen while:
   - Starting the application
   - Uploading a research paper
   - Asking various questions
   - Showing the responses
2. Convert to GIF using a tool like [ScreenToGif](https://www.screentogif.com/)
3. Save as `demo.gif` in your project root
4. Reference it in the README as shown above

Would you like me to help with creating the demo video or make any adjustments to the README?
