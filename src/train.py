import torch
from dataset import CharDataset, get_batch
from model import NanoGPT

def main():
    # 1. Hyperparameters Configuration
    batch_size = 32        # Number of parallel sequences processed together
    block_size = 64        # Context window length (characters)
    max_iters = 1500       # Total number of training steps
    learning_rate = 1e-3   # Step size for the optimizer (0.001)
    eval_interval = 200    # How often to check validation loss
    
    # Check for hardware acceleration (use GPU if available, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using execution device backing: {device}")

    # 2. Extract & Load Data Assets
    dataset = CharDataset('data/input.txt')
    vocab_size = dataset.vocab_size
    all_data = dataset.get_data_tensor()
    
    # Split dataset: 90% for training parameters, 10% for evaluation testing
    n = int(0.9 * len(all_data))
    train_data = all_data[:n]
    val_data = all_data[n:]

    # 3. Initialize Model and Optimization Routine
    model = NanoGPT(vocab_size=vocab_size, block_size=block_size)
    model.to(device) # Move model weights to GPU/CPU memory
    
    # Using AdamW (Standard high-performance optimizer for transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Beginning training loop optimization routine...")
    
    # 4. Core Optimization Execution Loop
    for iteration in range(max_iters):
        
        # Every once in a while, evaluate the training vs validation losses
        if iteration % eval_interval == 0 or iteration == max_iters - 1:
            model.eval() # Set model to evaluation mode
            with torch.no_grad():
                # Evaluate a small slice of batches to track metrics
                x_tr, y_tr = get_batch(train_data, block_size, batch_size)
                x_val, y_val = get_batch(val_data, block_size, batch_size)
                
                x_tr, y_tr = x_tr.to(device), y_tr.to(device)
                x_val, y_val = x_val.to(device), y_val.to(device)
                
                _, loss_tr = model(x_tr, y_tr)
                _, loss_val = model(x_val, y_val)
                
            print(f"Step {iteration:4d}: Train Loss = {loss_tr.item():.4f} | Val Loss = {loss_val.item():.4f}")
            model.train() # Revert back to training mode

        # Fetch a fresh, random training batch
        xb, yb = get_batch(train_data, block_size, batch_size)
        xb, yb = xb.to(device), yb.to(device)

        # Forward optimization pass
        logits, loss = model(xb, yb)
        
        # Backward gradient recalculation pass
        optimizer.zero_grad(set_to_none=True) # Flush old gradient calculations
        loss.backward()                       # Compute raw gradients
        optimizer.step()                      # Update model weights

    # 5. Export and Save the Final Learned Parameters
    torch.save(model.state_dict(), 'models/nanogpt_weights.pt')
    print("Optimization finished successfully! Saved model weights to 'models/nanogpt_weights.pt'")

if __name__ == "__main__":
    main()