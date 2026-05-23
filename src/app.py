import streamlit as st
import torch
import os
import requests
from groq import Groq

# 1. Component Architecture Imports
from model import NanoGPT
from dataset import CharDataset

# 2. Setup (Kept completely unchanged)
# This finds the root directory of your project automatically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(root_dir, "data", "input.txt")

# Check if file exists to prevent crashing and give a clean UI message
if not os.path.exists(data_path):
    st.error(f"⚠️ Data file not found at: {data_path}. Please check your repository structure!")
    st.stop()

dataset = CharDataset(data_path)
encode, decode = dataset.encode, dataset.decode

st.set_page_config(page_title="Multimodal AI Suite", page_icon="🚀", layout="centered")

# --- Sidebar ---
st.sidebar.header("🔑 Authentication")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
hf_token = st.sidebar.text_input("Hugging Face Token", type="password")

# Only 2 tabs now
tab_text, tab_image = st.tabs(["📝 Text", "🎨 Image Studio"])

# =========================================================================
# TAB 1: TEXT GENERATION
# =========================================================================
with tab_text:
    st.subheader("Select Text Generation Engine")
    engine_mode = st.selectbox(
        "Choose Local or Cloud Brain:",
        ["The Bard (Custom Scratch-GPT)", "General Assistant (Llama 3.1 via Groq)"]
    )

    @st.cache_resource
    def load_transformer_model():
        # Uses the absolute root directory logic
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        weights_path = os.path.join(root_dir, "models", "nanogpt_weights.pt")
        
        if not os.path.exists(weights_path): return None
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        model = NanoGPT(vocab_size=dataset.vocab_size) 
        checkpoint = torch.load(weights_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device

    model_data = load_transformer_model()

    if engine_mode == "The Bard (Custom Scratch-GPT)":
        max_new_tokens = st.slider("Characters to Generate", 50, 1000, 300, 50)
        temperature = st.slider("Creativity", 0.1, 1.5, 1.0, 0.1)
        seed_text = st.text_input("Starting Prompt (Seed)", value="\n")
        
        if st.button("✨ Compose Shakespearean Script", type="primary"):
            if model_data is not None:
                model, device = model_data
                with st.spinner("Writing..."):
                    context_idx = torch.tensor([encode(seed_text)], dtype=torch.long, device=device)
                    generated_tokens = model.generate(context_idx, max_new_tokens=max_new_tokens, temperature=temperature)[0].tolist()
                    st.code(decode(generated_tokens), language="text")
            else:
                st.error("Weights file not found at models/nanogpt_weights.pt")

    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

        for msg in st.session_state.chat_history[1:]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if user_prompt := st.chat_input("Message the Assistant..."):
            if not groq_api_key:
                st.error("Please enter your Groq API key in the sidebar first!")
            else:
                with st.chat_message("user"): st.markdown(user_prompt)
                st.session_state.chat_history.append({"role": "user", "content": user_prompt})

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            client = Groq(api_key=groq_api_key)
                            completion = client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=st.session_state.chat_history,
                                temperature=0.7,
                            )
                            response_text = completion.choices[0].message.content
                            st.markdown(response_text)
                            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                        except Exception as e:
                            st.error(f"API Error: {e}")

# =========================================================================
# TAB 2: IMAGE STUDIO
# =========================================================================
with tab_image:
    st.subheader("🎨 AI Image Generator")
    image_prompt = st.text_area("Describe your image:")
    if st.button("🚀 Render Image"):
        if not hf_token: st.error("Need HF Token!")
        else:
            with st.spinner("Generating..."):
                try:
                    API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
                    headers = {"Authorization": f"Bearer {hf_token.strip()}"}
                    response = requests.post(API_URL, headers=headers, json={"inputs": image_prompt}, timeout=60)
                    if response.status_code == 200:
                        st.image(response.content)
                        st.download_button("📥 Download", response.content, "image.png")
                    else: st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e: st.error(f"Error: {e}")