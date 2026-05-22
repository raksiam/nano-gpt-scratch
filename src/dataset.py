import torch

class CharDataset:
    def __init__(self, filepath: str):
        # 1. Read the raw text file (Extract)
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        # 2. Extract the unique character vocabulary (Transform)
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        
        # 3. Build the primary key/lookup catalogs
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, s: str) -> list[int]:
        """Convert a raw string into a list of integers."""
        return [self.stoi[c] for c in s]
        
    def decode(self, l: list[int]) -> str:
        """Convert a list of integers back into a raw string."""
        return ''.join([self.itos[i] for i in l])
        
    def get_data_tensor(self):
        """Convert the entire raw corpus into a massive 1D integer vector."""
        return torch.tensor(self.encode(self.text), dtype=torch.long)

# if __name__ == "__main__":
#     # Test execution block to verify functionality
#     dataset = CharDataset('data/input.txt')
#     print(f"Total vocabulary size detected: {dataset.vocab_size} unique characters.")
    
#     sample_text = "Hello World"
#     encoded = dataset.encode(sample_text)
#     decoded = dataset.decode(encoded)
    
#     print(f"Original Text: '{sample_text}'")
#     print(f"Encoded Integer List: {encoded}")
#     print(f"Decoded Text Verification: '{decoded}'")

def get_batch(data, block_size: int, batch_size: int):
    """
    Generates a batch of inputs (X) and targets (Y) using a sliding window.
    X shape: (batch_size, block_size)
    Y shape: (batch_size, block_size)
    """
    # Grab random starting index positions throughout the dataset
    # We subtract block_size so we don't overflow past the end of the data array
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Slice a window of size 'block_size' for inputs, and offset by 1 for targets
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

if __name__ == "__main__":
    dataset = CharDataset('data/input.txt')
    print(f"Total vocabulary size detected: {dataset.vocab_size} unique characters.")
    
    # 1. Convert the entire text file into one massive 1D integer vector
    all_data = dataset.get_data_tensor()
    print(f"Dataset completely tokenized into tensor of shape: {all_data.shape}\n")
    
    # 2. Test slicing a batch of 2 examples, each with a context window of 8 tokens
    X, Y = get_batch(all_data, block_size=8, batch_size=2)
    
    print("--- BATCH VERIFICATION ---")
    print(f"Inputs (X) shape: {X.shape}")
    print(f"Targets (Y) shape: {Y.shape}")
    print(f"Raw Input Matrix (X):\n{X}")
    print(f"Raw Target Matrix (Y):\n{Y}")