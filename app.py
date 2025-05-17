from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time

app = Flask(__name__)
CORS(app)

# Health check endpoint
@app.route("/")
def health_check():
    return jsonify({
        "status": "ready",
        "message": "PDF Chatbot API is running",
        "endpoints": {
            "POST /ask": "Process PDF questions"
        }
    })

# Simple mock response for testing
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        # Mock response for testing
        return jsonify({
            "answer": f"Test response to: {question}",
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": "Service error",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses 10000
    app.run(host="0.0.0.0", port=port)
