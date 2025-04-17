import streamlit as st
import requests
import src.Utils as U
from src.HyperParameters import *
from src.Model import Network_Dection_Model
from src.Model_Predict import check_pipeline
import src.preprocessing_functions
import json
import numpy as np
from src.nids import IntrusionDetectionSystem
import threading

Model_Name = MODEL_NAME + '.pth'

### LOAD TRAINED MODEL ###
with open(U.model_params, 'r') as j:
    model_params = json.load(j)

ids = IntrusionDetectionSystem()
ids_thread = threading.Thread(target=ids.start(), daemon=True)
ids_thread.start()
### Finished loading trained model ###

temperature = 1

url = "http://localhost:5001/v1/chat/completions"  # Replace with your KoboldCPP API URL
headers = {"Content-Type": "application/json"}

payload = {
    "model": "your-model-name",
    "messages": [
            
        ],
    "temperature": temperature,
    "max_tokens": 1000
}

'''# Load the DialoGPT model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # You can replace with the desired model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)'''

def chat_with_ai(payload):
    response = requests.post(url, headers=headers, json=payload)
    response = response.json()

    assistant_response = response["choices"][0]["message"]["content"]
    return assistant_response

# Create the text-generation pipeline
#pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# STREAMLIT APP #
# Initialize the chat history if it doesn't exist in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("LLM Chat")
st.write("Chat with me!")

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("What's on your mind?"):
    payload['messages'].append({"role": "user", "content": prompt})
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using the pipeline
    # Build conversation context for the model by joining previous user and assistant messages
    conversation = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]) + "\nAssistant:"
    
    # Generate response from the model
    #outputs = pipe(conversation, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    bot_response = chat_with_ai(payload)
    payload['messages'].append({"role": "assistant", "content": bot_response})
    print(payload)

    #bot_response = outputs[0]["generated_text"].split("Assistant:")[-1].strip()

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    # Add LLM response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

