from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os

app = Flask(__name__)
CORS(app)

# Initialize components
vector_store = None
qa_chain = None

def initialize_components():
    global vector_store, qa_chain
    if vector_store is None:
        try:
            print("[INFO] Initializing components...")
            
            # Initialize LLM
            llm = ChatOpenAI(
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                model_name="deepseek/deepseek-chat-v3-0324:free",
                temperature=0.1
            )

            # Load and process PDF
            loader = PyPDFLoader("MKRCP.pdf")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            # Create QA chain
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
def home():
    return "✅ PDF Chatbot API is running"

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        if qa_chain is None:
            initialize_components()
            
        result = qa_chain({"query": question})
        return jsonify({"answer": result["result"]})

    except Exception as e:
        print(f"[ERROR] API error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    initialize_components()  # Pre-initialize on startup
    app.run(host="0.0.0.0", port=port, debug=False)
