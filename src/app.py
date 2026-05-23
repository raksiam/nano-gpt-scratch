import streamlit as st
import torch
import sys
import os
from groq import Groq  # Import the new AI engine

# 1. Component Architecture Imports
from model import NanoGPT
from dataset import CharDataset

# 2. Global Context & Vocabulary Pipeline Setup
data_path = os.path.join(os.path.dirname(__file__), "../data/input.txt")
dataset = CharDataset(data_path)

encode = dataset.encode
decode = dataset.decode

# --- Page Configuration ---
st.set_page_config(
    page_title="Hybrid AI Engine",
    page_icon="🤖",
    layout="centered"
)

# --- Application Header ---
st.title("🤖 Hybrid AI Engine")
st.write("Switch between a custom scratch-built Shakespeare model and a powerful modern assistant.")

# --- Sidebar Configuration Panels ---
st.sidebar.header("⚙️ Engine Selection")

# Dropdown choice for the model backend
engine_mode = st.sidebar.selectbox(
    "Choose AI Brain:",
    ["The Bard (Custom Scratch-GPT)", "General Assistant (Llama 3.1 via Groq)"]
)

st.sidebar.markdown("---")
st.sidebar.header("🔑 API Authentication")
# 🌟 GLOBAL FIX: Defined ONCE here so it works across both engine views without duplicates!
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password", placeholder="gsk_...")

# Context-aware settings depending on selection
if engine_mode == "The Bard (Custom Scratch-GPT)":
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Generation Settings")
    max_new_tokens = st.sidebar.slider("Characters to Generate", min_value=50, max_value=1000, value=300, step=50)
    temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.1, max_value=1.5, value=1.0, step=0.1)
    seed_text = st.sidebar.text_input("Starting Prompt (Seed)", value="\n")

# --- Model Loading Loop for Custom GPT ---
@st.cache_resource
def load_transformer_model():
    weights_path = os.path.join(os.path.dirname(__file__), "../models/nanogpt_weights.pt")
    if not os.path.exists(weights_path):
        return None
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = NanoGPT(vocab_size=dataset.vocab_size) 
    checkpoint = torch.load(weights_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, device

model_data = load_transformer_model()

# --- Execution Logic ---

# MODE 1: CUSTOM SHAKESPEARE GPT
if engine_mode == "The Bard (Custom Scratch-GPT)":
    if model_data is not None:
        model, device = model_data
        if st.button("✨ Compose Shakespearean Script", type="primary"):
            with st.spinner("The Bard is thinking..."):
                context_idx = torch.tensor([encode(seed_text)], dtype=torch.long, device=device)
                generated_tokens = model.generate(context_idx, max_new_tokens=max_new_tokens, temperature=temperature)[0].tolist()
                st.success("Generation Complete!")
                st.code(decode(generated_tokens), language="text")

# MODE 2: GENERAL ASSISTANT (Continuous Conversation Engine)
else:
    # 1. Initialize conversation history container in memory if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful, brilliant daily assistant capable of holding long conversations."}
        ]

    # 2. Render all past messages to the screen to maintain the chat app look
    for msg in st.session_state.chat_history[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. Create a real-time chat input bar at the bottom of the screen
    if user_prompt := st.chat_input("Message the Assistant..."):
        
        # Pulls the API key seamlessly from the global sidebar variable
        if not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar first!")
        else:
            # Render user input instantly to the UI
            with st.chat_message("user"):
                st.markdown(user_prompt)
                
            # Append the user's new question into persistent memory
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})

            # Call the AI model with the ENTIRE message history chain
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = Groq(api_key=groq_api_key)
                        
                        # Pass the full history array instead of just a single prompt
                        completion = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=st.session_state.chat_history,
                            temperature=0.7,
                        )
                        
                        response_text = completion.choices[0].message.content
                        st.markdown(response_text)
                        
                        # Append the model's response back into persistent memory
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                        
                    except Exception as e:
                        st.error(f"API Error: {e}")

    # Add a reset button to the sidebar to clear memory if the user wants a fresh start
    if st.sidebar.button("🗑️ Clear Chat History"):
        del st.session_state.chat_history
        st.rerun()