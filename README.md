# NanoGPT: Character-Level Shakespeare Language Model

An end-to-end character-level Generative Transformer model built completely from scratch in PyTorch, inspired by the NanoGPT architecture. This project implements a fully functional data processing pipeline, a custom causal self-attention mechanism, an optimization training loop, and an interactive text generation inference script.

## 📝 Project Overview

This project builds a generative language model trained on the text of Shakespeare's plays. Instead of working with whole words, the model reads, processes, and generates text character-by-character. By building the network blocks from scratch, the project demonstrates how data pipelines feed into a causal self-attention system, how optimization loops minimize structural entropy, and how multinomial sampling processes generate completely original text mimicking a target style.

---

## 📂 Project Repository Structure
.
├── data/
│   └── input.txt             # Raw text corpus (Tiny Shakespeare)
├── models/
│   └── nanogpt_weights.pt    # Saved optimized model weights
├── src/
│   ├── dataset.py            # Data loading, tokenization, and batching logic
│   ├── model.py              # Custom Transformer architecture and layers
│   ├── train.py              # Training loop optimization routine
│   └── generate.py           # Text generation inference runner
├── .gitignore                # Excludes caches and model binaries
└── README.md                 # Project documentation

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

Follow these exact steps from your terminal to install dependencies, train the model, and generate original Shakespearean text.

### 1. Environment Activation & Dependency Installation

Navigate to your project root, create a clean virtual environment container, and install PyTorch:

```bash
# Navigate to project folder
cd nano-gpt-scratch

# Initialize and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, run: venv\\Scripts\\activate

# Install the machine learning library dependency
pip install torch

```

### 2. Run the Training Optimization Loop

To train the model from scratch on the text corpus and export the learned weights file, run:

```bash
python3 src/train.py

```

*The terminal will output current training and validation loss calculations every 200 cycles as the network learns patterns.*

### 3. Generate Original Text (Inference)

Once training is complete and `models/nanogpt_weights.pt` is generated, run the text generation utility to watch the model write original dialogue:

```bash
python3 src/generate.py

```

---

## 📊 Expected Output & Evaluation

* **Baseline Loss:** ~4.17 - 4.60 (Equivalent to uniform random guessing across the 65-character vocabulary layout).
* **Final Target Loss:** < 2.00 (Demonstrating structural learning and lower text entropy).

### Sample Model Generation:

```text
MORKENTMBETH:
And your can blood to my mast, afferch I sooft
Draw colive I was tell in love struck, not
And none in so helpar of ye heard?
Orife, good woodst Geord were the chook you the wort

```

"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)
print("File README.md created successfully.")

```
Your Markdown file is ready
[file-tag: code-generated-file-0-1779544647567052210]

The `README.md` file has been generated with all your requested information. It includes:
1. **Project Overview** explaining what the character-level language model is.
2. **Project Repository Structure** detailing where files like `src/train.py`, `src/model.py`, and `data/input.txt` live.
3. **Summary of Steps Followed** outlining your journey through setup, modeling, training, and setting up your GitHub Personal Access Token/credential helper settings.
4. **How to Run the Project** with explicit step-by-step terminal commands for virtual environments, dependency installation, training execution, and generation tracking.
5. **Expected Output & Metrics** evaluating baseline random entropy versus optimization results with a final sample text generation snippet.

```
