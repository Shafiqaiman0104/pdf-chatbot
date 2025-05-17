from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Changed import
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os
from typing import Dict, Any

app = Flask(__name__)
CORS(app)

# Initialize LLM (DeepSeek via OpenRouter)
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name="deepseek/deepseek-chat-v3-0324:free",
    temperature=0.1
)

# Cache components
vector_store = None
qa_chain = None

def initialize_components():
    global vector_store, qa_chain
    if vector_store is None:
        try:
            print("[INFO] Loading and processing PDF...")
            loader = PyPDFLoader("MKRCP.pdf")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever()
            )
            print("[INFO] Components initialized successfully")
        except Exception as e:
            print(f"[ERROR] Initialization failed: {str(e)}")
            raise

@app.route("/")
def home() -> str:
    return "✅ PDF Chatbot API is running"

@app.route("/ask", methods=["POST"])
def ask() -> Dict[str, Any]:
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return {"error": "Question cannot be empty"}, 400

        if qa_chain is None:
            initialize_components()
            
        result = qa_chain({"query": question})
        return {"answer": result["result"]}

    except Exception as e:
        print(f"[ERROR] API error: {str(e)}")
        return {"error": "Internal server error"}, 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    initialize_components()  # Pre-initialize on startup
    app.run(host="0.0.0.0", port=port, debug=False)
