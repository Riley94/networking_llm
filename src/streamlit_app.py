import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the DialoGPT model and tokenizer
model_name = "openai-community/gpt2"  # You can replace with the desired model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create the text-generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

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
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using the pipeline
    # Build conversation context for the model by joining previous user and assistant messages
    conversation = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]) + "\nAssistant:"
    
    # Generate response from the model
    outputs = pipe(conversation, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    bot_response = outputs[0]["generated_text"].split("Assistant:")[-1].strip()

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    # Add LLM response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

