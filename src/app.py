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

# 🚨 THE NEW DROPDOWN METER
engine_mode = st.sidebar.selectbox(
    "Choose AI Brain:",
    ["The Bard (Custom Scratch-GPT)", "General Assistant (Llama 3 via Groq)"]
)

# Context-aware settings depending on selection
if engine_mode == "The Bard (Custom Scratch-GPT)":
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Generation Settings")
    max_new_tokens = st.sidebar.slider("Characters to Generate", min_value=50, max_value=1000, value=300, step=50)
    temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.1, max_value=1.5, value=1.0, step=0.1)
    seed_text = st.sidebar.text_input("Starting Prompt (Seed)", value="\n")
else:
    st.sidebar.markdown("---")
    st.sidebar.header("🔑 API Authentication")
    # Users can safely paste their free Groq key here
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password", placeholder="gsk_...")
    user_prompt = st.text_area("Ask the Assistant anything (Day-to-day questions, coding, advice, etc.):", placeholder="Why is the sky blue?")

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

# MODE 2: GENERAL ASSISTANT (Day-to-day Questions)
else:
    if st.button("🚀 Ask Assistant", type="primary"):
        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar first!")
        elif not user_prompt.strip():
            st.warning("Please type a question first!")
        else:
            with st.spinner("Consulting Llama-3..."):
                try:
                    # Initialize Groq client with user's key
                    client = Groq(api_key=groq_api_key)
                    
                    # Request a chat completion from a state-of-the-art model
                    completion = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": "You are a helpful, brilliant daily assistant."},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                    )
                    
                    # Output response nicely formatted in markdown
                    st.success("Response:")
                    st.markdown(completion.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"API Error: {e}")