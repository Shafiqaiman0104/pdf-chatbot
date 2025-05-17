from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Setup LLM (DeepSeek via OpenRouter)
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name="deepseek/deepseek-chat-v3-0324:free",
)

# Cache the QA chain so we don’t reload every time
qa_chain = None

def get_qa_chain():
    global qa_chain
    if qa_chain is None:
        print("[INFO] Loading and splitting PDF...")
        loader = PyPDFLoader("MKRCP.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embedding)

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        print("[INFO] QA chain ready.")
    return qa_chain

# Health check route
@app.route("/")
def home():
    return "✅ PDF Chatbot is running."

# PDF Q&A route
@app.route("/ask", methods=["POST"])
def ask():
    try:
        print("[INFO] Received /ask request")
        data = request.get_json()
        question = data.get("question", "")
        print(f"[INFO] Question: {question}")

        if not question:
            return jsonify({"error": "Question is required"}), 400

        qa = get_qa_chain()
        answer = qa.run(question)
        print(f"[INFO] Answer: {answer}")
        return jsonify({"answer": answer})

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
