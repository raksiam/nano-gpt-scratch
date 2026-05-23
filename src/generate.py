import torch
from dataset import CharDataset
from model import NanoGPT

def main():
    # 1. Initialize our token mapping catalog
    dataset = CharDataset('data/input.txt')
    
    # 2. Reconstruct the blank model architecture framework
    model = NanoGPT(vocab_size=dataset.vocab_size)
    
    # 3. Stream in our saved learned weight values from disk
    # (weights are mapped automatically to CPU execution context for generation stability)
    model.load_state_dict(torch.load('models/nanogpt_weights.pt', map_location='cpu'))
    model.eval() # Set to evaluation mode to shut off internal scaling modules
    
    # 4. Initialize an empty starting token canvas (index 0 representing a newline char)
    # This acts as the seed from which our model starts generating
    context = torch.zeros((1, 1), dtype=torch.long)
    
    print("\n--- GENERATING 500 CHARACTERS OF FRESH SHAKESPEARE ---")
    
    # Run the generator function for 500 characters
    generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
    
    # Convert integer indices back into characters and print to screen
    output_text = dataset.decode(generated_indices)
    print(output_text)
    print("\n-----------------------------------------------------")

if __name__ == "__main__":
    main()