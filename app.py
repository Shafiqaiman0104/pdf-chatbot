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

# Ensure PDF exists
if not os.path.exists("MKRCP.pdf"):
    raise FileNotFoundError("The file MKRCP.pdf was not found. Make sure it's uploaded to your repo.")

# Load and split PDF
loader = PyPDFLoader("MKRCP.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Create vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embedding)

# Setup LLM (DeepSeek via OpenRouter)
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name="deepseek/deepseek-chat-v3-0324:free",
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    answer = qa_chain.run(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
