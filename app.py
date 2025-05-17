from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# Initialize LLM (DeepSeek via OpenRouter)
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name="deepseek/deepseek-chat-v3-0324:free",
    temperature=0.1
)

# Cache QA chain
qa_chain = None

def get_qa_chain():
    global qa_chain
    if qa_chain is None:
        try:
            print("[INFO] Loading and splitting PDF...")
            loader = PyPDFLoader("MKRCP.pdf")
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(docs)

            embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Simplified Chroma initialization
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            print("[INFO] QA chain initialized successfully.")
        except Exception as e:
            print("[ERROR] QA chain initialization failed:", str(e))
            raise
    return qa_chain

@app.route("/")
def home():
    return "✅ PDF Chatbot API is running."

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        qa = get_qa_chain()
        result = qa({"query": question})
        return jsonify({"answer": result["result"]})

    except Exception as e:
        print("[ERROR] API error:", str(e))
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
