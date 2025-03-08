from flask import Flask, request, jsonify, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import datetime

app = Flask(__name__)

# Load the DialoGPT model and tokenizer
model_name = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create the text-generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Store chat history per user session
chat_histories = {}

# Simple HTML UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LLM Chat</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        textarea { width: 80%; height: 100px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        #response { margin-top: 20px; font-size: 18px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Interact with LLM</h1>
    <textarea id="question" placeholder="Enter your message..."></textarea><br><br>
    <button onclick="generateResponse()">Ask</button>
    <div id="response"></div>

    <script>
        async function generateResponse() {
            let question = document.getElementById("question").value;
            let responseDiv = document.getElementById("response");
            
            responseDiv.innerHTML = "Thinking...";

            let response = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question })
            });

            let data = await response.json();
            responseDiv.innerHTML = data.answer;
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/generate", methods=["POST"])
def generate():
    """Handles chatbot-style queries using GPT-2"""
    data = request.json
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "A question is required"}), 400

    user_id = request.remote_addr  # Use IP as a basic session ID
    if user_id not in chat_histories:
        chat_histories[user_id] = []

    # Append the user query
    chat_histories[user_id].append(f"User: {question}")

    # Format conversation history into a single string
    conversation = "\n".join(chat_histories[user_id]) + "\nBot:"

    # Generate response
    outputs = pipe(conversation, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    
    # Extract generated response
    response_text = outputs[0]["generated_text"].split("Bot:")[-1].strip()

    # Append chatbot response to history
    chat_histories[user_id].append(f"Bot: {response_text}")

    return jsonify({"answer": response_text})


@app.route("/analyze_alert", methods=["POST"])
def analyze_alert():
    """Handles alert analysis with conversational context"""
    data = request.json
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "A question is required"}), 400

    user_id = request.remote_addr  # Track conversations per user
    if user_id not in chat_histories:
        chat_histories[user_id] = [
            {"role": "system", "content": "You are a security alert analysis assistant."}
        ]

    # Append user query
    chat_histories[user_id].append({"role": "user", "content": question})

    # Generate response
    outputs = pipe(chat_histories[user_id], max_new_tokens=512)
    
    # Extract generated response
    response_text = outputs[0]["generated_text"][-1]["content"]

    # Append chatbot response to history
    chat_histories[user_id].append({"role": "assistant", "content": response_text})

    return jsonify({
        "analysis": response_text,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)