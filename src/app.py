import streamlit as st
import torch
import sys
import os

from model import NanoGPT
from dataset import CharDataset  # import the CharDataset class from dataset.py

# Setup the Dataset FIRST (so 'dataset' exists globally)
data_path = os.path.join(os.path.dirname(__file__), "../data/input.txt")
dataset = CharDataset(data_path)

# Extract your helper functions
encode = dataset.encode
decode = dataset.decode

# --- Page Configuration ---
st.set_page_config(
    page_title="NanoGPT Shakespeare Generator",
    page_icon="🎭",
    layout="centered"
)

# --- Application Header ---
st.title("🎭 NanoGPT Creative Engine")
st.subheader("Generate custom Shakespearean dialogue from your scratch-built Transformer")
st.write("Adjust the generation properties in the sidebar and watch the model compose text character-by-character.")

# --- Sidebar Configuration Panels ---
st.sidebar.header("🛠️ Generation Settings")

# User inputs to control creativity
max_new_tokens = st.sidebar.slider("Characters to Generate", min_value=50, max_value=1000, value=300, step=50)
temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.1, max_value=1.5, value=1.0, step=0.1)
seed_text = st.sidebar.text_input("Starting Prompt (Seed)", value="\n")

# --- Model Loading Loop (Cached so it only runs once) ---
@st.cache_resource
def load_transformer_model():
    weights_path = os.path.join(os.path.dirname(__file__), "../models/nanogpt_weights.pt")
    
    # Check if weights file exists
    if not os.path.exists(weights_path):
        st.error(f"Could not find model weights at `{weights_path}`. Please run training first!")
        return None

    # Detect execution hardware
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Initialize your model architecture matching your train parameters
    # Note: Replace these placeholder numbers with your model's exact hyper-parameters
    model = NanoGPT(vocab_size=dataset.vocab_size)
    
    # Load learned weights safely across devices
    checkpoint = torch.load(weights_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, device

# Initialize model
model_data = load_transformer_model()

# --- Text Generation Trigger ---
if model_data is not None:
    model, device = model_data

    if st.button("✨ Compose Shakespearean Script", type="primary"):
        with st.spinner("The Bard is thinking..."):
            try:
                # Convert string prompt into numeric tensor sequence
                context_idx = torch.tensor([encode(seed_text)], dtype=torch.long, device=device)
                
                # Run inference through the network 
                # (Ensure your model's generate function accepts max_new_tokens/temperature)
                generated_tokens = model.generate(context_idx, max_new_tokens=max_new_tokens, temperature=temperature)[0].tolist()
                
                # Transform tokens back to readable string text
                result_text = decode(generated_tokens)
                
                # Display output inside a clean web code block container
                st.success("Generation Complete!")
                st.code(result_text, language="text")
                
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")