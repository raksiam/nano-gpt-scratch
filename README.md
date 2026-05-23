# 🤖 Hybrid AI Engine: Custom NanoGPT & General Assistant

An end-to-end multi-engine AI application built in PyTorch and Streamlit. This project showcases two distinct layers of AI development: a character-level Generative Transformer model built completely from scratch, paired side-by-side with a modern open-source Large Language Model (Llama 3.1) via the Groq cloud API to handle everyday natural language questions.

## 📝 Project Overview

This project serves as a comprehensive portfolio bridge between foundational deep learning architecture and production-ready cloud LLM integration. 

1. **The Bard Engine (Custom Scratch-GPT):** A generative language model trained on the text of Shakespeare's plays. Reading, processing, and generating text character-by-character, it demonstrates how data pipelines feed into a causal self-attention system, how optimization loops minimize structural entropy, and how multinomial sampling processes generate completely original stylized text.
2. **The Assistant Engine (Production API Integration):** A state-of-the-art conversational layout that taps into the high-speed Groq cloud architecture to solve general tasks, coding, and day-to-day text execution workflows.

---

## 📂 Project Repository Structure

* **nano-gpt-scratch/**
    * 📁 **data/**
        * 📄 `input.txt`
    * 📁 **models/**
        * 📄 `nanogpt_weights.pt`
    * 📁 **src/**
        * 📄 `app.py` — Hybrid web application dashboard script
        * 📄 `dataset.py`
        * 📄 `generate.py`
        * 📄 `model.py`
        * 📄 `train.py`
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
6. **Cloud Infrastructure Security:** Attached the local tree to a GitHub remote destination using an authenticated Personal Access Token (PAT) paired with global OS credential keychain helpers for automated future pushes.

---

## 🚀 How to Run the Project

Follow these exact steps from your terminal to install dependencies, train the model, and generate original text.

### 1. Environment Activation & Dependency Installation

Navigate to your project root, create a clean virtual environment container, and install PyTorch and the required web/API SDK libraries:

```bash
# Navigate to project folder
cd nano-gpt-scratch

# Initialize and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, run: venv\Scripts\activate

# Install machine learning, UI, and external AI SDK dependencies
pip install torch streamlit groq

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

### 4. Interactive Hybrid Web Dashboard UI

The application features an interactive Streamlit graphical interface (`src/app.py`) that serves as a toggleable dual-engine platform.

#### Key Framework Implementations:

* **Dynamic Vocabulary Loading:** Instantiates the custom `CharDataset` pipeline first to dynamically feed `vocab_size` directly into the `NanoGPT` model constructor, eliminating hardcoded array dimension mismatch errors.
* **Cross-Hardware Weight Deserialization:** Utilizes PyTorch's `map_location` parameter to safely stream learned checkpoint matrices smoothly across differing backend target environments (CUDA, MPS, or CPU).
* **Multi-Engine Intelligence System:** Adds a runtime router allowing users to switch between the local custom Transformer (with adjustable temperature randomness scaling) and a fast cloud connection executing the `llama-3.1-8b-instant` assistant layer.

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

### Sample Assistant Output:

```text
User Question: "Give me a simple explanation of Python Decorators"
Response: "A decorator in Python is a design pattern that allows you to modify or extend the behavior of a function or method without permanently altering its source code. Think of it like wrapping a gift—the decorator is the wrapping paper that changes the outside appearance/behavior, while the gift inside remains identical!"

```

```
