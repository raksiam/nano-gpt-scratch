import streamlit as st
import torch
import os
import requests
from groq import Groq

# 1. Component Architecture Imports
from model import NanoGPT
from dataset import CharDataset

# 2. Setup (Absolute repository paths for cloud deployment)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(root_dir, "data", "input.txt")

if not os.path.exists(data_path):
    st.error(f"⚠️ Data file not found at: {data_path}. Please check your repository structure!")
    st.stop()

dataset = CharDataset(data_path)
encode, decode = dataset.encode, dataset.decode

st.set_page_config(page_title="Multimodal AI Suite", page_icon="🚀", layout="centered")

# Track active tab via session state to correctly route the bottom global input box
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📝 Text Engine"

# --- Sidebar ---
st.sidebar.header("🔑 Authentication")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
hf_token = st.sidebar.text_input("Hugging Face Token", type="password")

# Render layout tabs
tab_text, tab_image = st.tabs(["📝 Text Engine", "🎨 Image Studio"])

# =========================================================================
# TAB 1: TEXT ENGINE
# =========================================================================
with tab_text:
    st.subheader("Select Text Generation Engine")
    engine_mode = st.selectbox(
        "Choose Local or Cloud Brain:",
        ["The Bard (Custom Scratch-GPT)", "General Assistant (Llama 3.1 via Groq)"]
    )

    @st.cache_resource
    def load_transformer_model():
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
        bard_output_area = st.container()
    else:
        chat_output_area = st.container()
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

        with chat_output_area:
            for msg in st.session_state.chat_history[1:]:
                with st.chat_message(msg["role"]): 
                    st.markdown(msg["content"])

# =========================================================================
# TAB 2: IMAGE STUDIO
# =========================================================================
with tab_image:
    st.subheader("🎨 AI Image Studio")
    image_output_area = st.container()

# CSS tweak to add bottom padding to prevent content from getting cut off behind the sticky input
st.markdown("<style>.stChatMessageContainer, .stBlock { padding-bottom: 70px; }</style>", unsafe_allow_html=True)

# =========================================================================
# 🚀 GLOBAL BOTTOM FOOTER INPUT (Reverted to Router URL)
# =========================================================================
universal_placeholder = "Type here..."

if user_prompt := st.chat_input(universal_placeholder):
    
    # Check if user is trying to run an image generation based on prompt/tokens
    image_keywords = ["draw", "image", "generate", "paint", "photo", "picture", "flux", "art", "cinematic", "shot", "rendering", "style"]
    
    if (hf_token and not groq_api_key) or any(word in user_prompt.lower() for word in image_keywords):
        is_image_job = True
    else:
        is_image_job = False

    # --- EXECUTION ROUTING ---
    if is_image_job:
        with image_output_area:
            if not hf_token: 
                st.error("Please enter your Hugging Face Token in the sidebar to generate images!")
            else:
                with st.spinner("Generating image via FLUX.1..."):
                    try:
                        # REVERTED: Restored your original working cloud-router path
                        API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
                        headers = {"Authorization": f"Bearer {hf_token.strip()}"}
                        
                        response = requests.post(API_URL, headers=headers, json={"inputs": user_prompt}, timeout=60)
                        
                        if response.status_code == 200:
                            st.image(response.content, caption=f"Prompt: {user_prompt}")
                            st.download_button("📥 Download Render", response.content, "image.png")
                        elif response.status_code == 503:
                            st.warning("Hugging Face server is warming up the FLUX engine. Please wait 15 seconds and press enter again!")
                        else: 
                            st.error(f"Hugging Face API Error ({response.status_code}): {response.text}")
                            
                    except Exception as e: 
                        st.error(f"Error handling image workflow: {e}")
    else:
        # Route to Text Engine
        if engine_mode == "The Bard (Custom Scratch-GPT)":
            with bard_output_area:
                if model_data is not None:
                    model, device = model_data
                    with st.spinner("Composing Shakespearean script..."):
                        context_idx = torch.tensor([encode(user_prompt)], dtype=torch.long, device=device)
                        generated_tokens = model.generate(context_idx, max_new_tokens=max_new_tokens, temperature=temperature)[0].tolist()
                        st.text_area("The Bard Says:", value=decode(generated_tokens), height=300)
                else:
                    st.error("Weights file not found at models/nanogpt_weights.pt")
        
        elif engine_mode == "General Assistant (Llama 3.1 via Groq)":
            if not groq_api_key:
                with chat_output_area: 
                    st.error("Please enter your Groq API key in the sidebar first!")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_prompt})
                with chat_output_area:
                    with st.chat_message("user"): 
                        st.markdown(user_prompt)
                    
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