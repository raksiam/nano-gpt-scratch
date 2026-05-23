# 🤖 Multimodal Multitask AI Suite: Custom Transformer & Creative Studios

An end-to-end multimodal AI application built entirely in PyTorch and Streamlit. This comprehensive suite bridges the gap between foundational architectural deep learning and production-ready serverless cloud APIs. The system handles custom character-level text synthesis entirely on local consumer hardware, alongside dedicated pipelines for continuous assistant workflows and cloud-accelerated visual media generation.

## 📝 Project Overview

This project serves as an engineering portfolio that scales raw tensor logic into a modern, production-grade application across three specialized focus areas:

1. **The Bard Engine (Custom Scratch-GPT):** A generative language model trained on the text of Shakespeare's plays. Reading, processing, and generating text character-by-character, it demonstrates how raw token data pipelines feed into a causal self-attention system, how optimization loops minimize structural entropy, and how multinomial sampling processes generate completely original stylized text.
2. **The Assistant Engine (Continuous Conversation):** A state-of-the-art chat platform tapping into high-speed Groq cloud infrastructure. By wrapping the stateless API within an stateful memory buffer, it retains situational context for complex, multi-turn technical chats.
3. **The Image Studio (Multimodal Synthesis):** A creative studio that routes prompts to high-compute serverless cloud clusters hosting the state-of-the-art 12-Billion parameter `FLUX.1-schnell` model, rendering and exporting high-resolution PNG matrices.

---

## 📂 Project Repository Structure

* **nano-gpt-scratch/**
    * 📁 **data/**
        * 📄 `input.txt` — Raw training corpus (The Complete Works of Shakespeare)
    * 📁 **models/**
        * 📄 `nanogpt_weights.pt` — Compiled local model matrix checkpoints
    * 📁 **src/**
        * 📄 `app.py` — Main Streamlit Multimodal dashboard and orchestration interface
        * 📄 `dataset.py` — Custom text tokenization engine and batch array builder
        * 📄 `generate.py` — Command Line Interface inference routine
        * 📄 `model.py` — Custom PyTorch implementation of Multi-Head Attention blocks
        * 📄 `train.py` — Optimization loop and gradient calculation setup
    * 📄 `.gitignore`
    * 📄 `README.md`

---

## 🛠️ Summary of Steps Followed

During development, the project advanced through the following engineering milestones:

1. **Environment & Architecture Setup:** Initialized a local Git repository, created a structured modular file layout, and configured `.gitignore` parameters to ignore runtime caches and large model binaries.
2. **Data Pipeline Engineering (dataset.py):** Constructed a vocabulary mapper for 65 unique characters, created numerical array encoders/decoders, set up a 90/10 train/validation split, and built a batch generator using a window context window.
3. **Model Construction (model.py):** Wrote custom PyTorch layers implementing multi-head causal self-attention, masked lower-triangular tracking tables (tril), feed-forward expansion blocks, layer normalization, and residual skip-connections.
4. **Optimization Loop Integration (train.py):** Built an automated routine using the AdamW optimizer, added device fallback matching (CUDA/MPS/CPU), evaluated initial untrained entropy thresholds, and targeted validation monitoring over 1,500 steps.
5. **Inference Engine Deployment (generate.py):** Programmed a creative generation pipeline using a softmax probability distribution and random multinomial index sampling to yield original character strings.
6. **Stateful Conversation Memory Architecture:** Built an abstract state tracking system inside the dashboard to preserve text logs across multi-turn prompts, converting a stateless API into an interactive chat app.
7. **Multimodal Router Integration:** Designed a backend requests framework that handles token isolation and error timeouts, directly querying live serverless inference endpoints to render binary image arrays instantly.

---

## 🚀 How to Run the Project

Follow these exact steps from your terminal to install dependencies, train the model, and launch the web interface.

### 1. Environment Activation & Dependency Installation

Navigate to your project root, create a clean virtual environment container, and install the library requirements:

```bash
# Navigate to project folder
cd nano-gpt-scratch

# Initialize and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, run: venv\Scripts\activate

# Install machine learning, UI networking, and external AI SDK dependencies
pip install torch streamlit groq requests

```

### 2. Run the Training Optimization Loop

To train the custom model from scratch on the Shakespeare text corpus and export the learned weights file, run:

```bash
python3 src/train.py

```

*The terminal will output current training and validation loss calculations every 200 cycles as the network learns patterns.*

### 3. Generate Original Text (CLI Inference)

Once training is complete and `models/nanogpt_weights.pt` is generated, you can run the terminal-based text generation utility:

```bash
python3 src/generate.py

```

---

### 4. Interactive Multimodal Dashboard Dashboard

The application features a fully responsive, tabbed interface (`src/app.py`) to easily swap between local code blocks and cloud generation suites.

#### Key UI & Pipeline Implementations:

* **Dynamic Vocabulary Loading:** Instantiates the custom `CharDataset` pipeline first to dynamically feed `vocab_size` directly into the `NanoGPT` model constructor, eliminating hardcoded array dimension mismatch errors.
* **Cross-Hardware Weight Deserialization:** Utilizes PyTorch's `map_location` parameter to safely stream learned checkpoint matrices smoothly across differing backend target environments (CUDA, MPS, or CPU).
* **Global Authentication Management:** Consolidates access keys to the sidebar header, instantiating credentials single-turn to eliminate duplicate widget element key errors across script re-runs.
* **Serverless Image Studio:** Integrates secure HTTP post request layers to communicate with the active Hugging Face Serverless Router (`router.huggingface.co`), parsing raw response bytes into downloadable, high-resolution visual formats on the fly.

To launch the local web server dashboard, execute:

```bash
streamlit run src/app.py

```

---

## 📊 Expected Output & Evaluation

* **Baseline Loss:** ~4.17 - 4.60 (Equivalent to uniform random guessing across the 65-character vocabulary layout).
* **Final Target Loss:** < 2.00 (Demonstrating structural learning and lower text entropy).

### Sample Custom Transformer Output:

```text
MORKENTMBETH:
And your can blood to my mast, afferch I sooft
Draw colive I was tell in love struck, not
And none in so helpar of ye heard?
Orife, good woodst Geord were the chook you the wort

```

### Sample Conversational Assistant Output:

```text
User: Hi, I'm developing an AI suite! Remember my name, okay?
Assistant: That sounds like an awesome project! I've noted down your name. How can I help you build it?

User: What is my name and what am I building?
Assistant: Your name is John, and you are building a Multimodal AI Suite that includes a custom scratch-built Shakespeare Transformer!

```
