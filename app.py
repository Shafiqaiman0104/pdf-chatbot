from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

app = Flask(__name__)
CORS(app)

# Global variables
model = None
pdf_texts = []
embeddings = []

def load_components():
    global model, pdf_texts, embeddings
    
    # 1. Load lightweight sentence model
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # 2. Process PDF
    with open("MKRCP.pdf", "rb") as f:
        reader = PdfReader(f)
        pdf_texts = [page.extract_text() for page in reader.pages if page.extract_text()]
        
        # Split into smaller chunks (better for search)
        text_chunks = []
        for text in pdf_texts:
            words = text.split()
            for i in range(0, len(words), 200):  # 200 words per chunk
                text_chunks.append(" ".join(words[i:i+200]))
        
        pdf_texts = text_chunks
    
    # 3. Create embeddings
    embeddings = model.encode(pdf_texts)

# Load components when starting (remove if too slow)
load_components()

@app.route("/")
def health_check():
    return "PDF Chatbot API is running"

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        # Get question embedding
        question_embed = model.encode([question])
        
        # Find most relevant text chunk
        similarities = cosine_similarity(question_embed, embeddings)
        best_match_idx = np.argmax(similarities)
        best_text = pdf_texts[best_match_idx]
        
        # Simple response (replace with LLM if needed)
        return jsonify({
            "answer": f"From the document: {best_text[:500]}...",  # Truncate long responses
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": "Error processing your question",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
