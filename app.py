from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Lighter than Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAIChat
import os
import time

app = Flask(__name__)
CORS(app)

# Lightweight model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "deepseek/deepseek-chat-v3-0324:free"
PDF_PATH = "MKRCP.pdf"

# Initialize with lazy loading
vector_store = None
qa_chain = None
last_init_time = 0

def initialize_components():
    global vector_store, qa_chain, last_init_time
    
    # Skip re-initialization if done recently
    if vector_store and (time.time() - last_init_time < 300):
        return
    
    try:
        print("[INIT] Loading lightweight components...")
        
        # 1. Use smaller text splitter
        text_splitter = CharacterTextSplitter(
            chunk_size=500,  # Reduced from 1000
            chunk_overlap=50
        )
        
        # 2. Process PDF in memory-efficient way
        with open(PDF_PATH, "rb") as f:
            from PyPDF2 import PdfReader
            pdf = PdfReader(f)
            text = "\n".join([page.extract_text() for page in pdf.pages])
            chunks = text_splitter.split_text(text)
        
        # 3. Use FAISS instead of Chroma
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        # 4. Initialize LLM
        llm = OpenAIChat(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model_name=LLM_MODEL,
            temperature=0.1,
            max_tokens=200  # Limit response length
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 2})  # Fewer chunks
        )
        
        last_init_time = time.time()
        print("[INIT] Components ready")
        
    except Exception as e:
        print(f"[ERROR] Initialization failed: {str(e)}")
        raise

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "Empty question"}), 400
            
        initialize_components()  # Lazy loading
        
        result = qa_chain({"query": question})
        return jsonify({
            "answer": result.get("result", "No answer found"),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": "Service unavailable",
            "details": str(e)[:100]  # Truncate long errors
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
