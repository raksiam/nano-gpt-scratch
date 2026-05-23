import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ One head of causal self-attention """
    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        # Linear projections to generate our math vectors
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Causal Masking: A lower-triangular matrix of ones.
        # This acts as a circuit breaker, preventing tokens from looking into future indices.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape # Batch, Time (Sequence Length), Channels (Embedding Size)
        
        # 1. Project inputs into Query, Key, and Value spaces
        k = self.key(x)   # Shape: (B, T, head_size)
        q = self.query(x) # Shape: (B, T, head_size)
        
        # 2. Compute attention scores (affinities between all tokens)
        # (B, T, head_size) @ (B, head_size, T) -> Shape: (B, T, T)
        # We scale by 1/sqrt(head_size) to keep variance stable at 1.0
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        
        # 3. Apply Causal Masking
        # Forces the model to only look at past or current tokens.
        # Future tokens get set to -infinity so they drop to 0% weight after Softmax.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # 4. Normalize rows into probabilities
        wei = F.softmax(wei, dim=-1) # Shape: (B, T, T)
        
        # 5. Aggregate the values based on calculated weights
        v = self.value(x) # Shape: (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) -> Shape: (B, T, head_size)
        
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of causal self-attention running in parallel """
    def __init__(self, num_heads: int, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        # Create a list containing independent instances of our Head class
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        # Projection layer to merge the concatenated head outputs back into the workspace dimension
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        # Run every head independently and concatenate their outputs along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project back to our base model embedding dimension
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            # Expand the internal hidden layer size by 4x (Standard Transformer design blueprint)
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # Project back down to model workspace dimension
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communicates (attention) then computes (feed-forward) """
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, n_embd, head_size, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 1. Communication Phase (with Residual skip connection and normalization)
        x = x + self.sa(self.ln1(x))
        # 2. Computation Phase (with Residual skip connection and normalization)
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int = 128, block_size: int = 64, n_head: int = 4, n_layer: int = 3):
        super().__init__()
        self.block_size = block_size
        
        # 1. Base Workspace Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # 2. Stacking multiple Transformer Blocks sequentially
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        
        # 3. Output prediction head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_arr = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos_arr)
        
        # Build workspace
        x = tok_emb + pos_emb
        
        # Pass through all Transformer Blocks!
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Output raw scores
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        """
        Takes an input matrix of token coordinates 'idx' (shape: B, T) 
        and generates 'max_new_tokens' characters of fresh text.
        """
        for _ in range(max_new_tokens):
            # Crop the current sequence context window if it exceeds our block_size limit
            # Since position embeddings only support dimensions up to block_size
            idx_cond = idx[:, -self.block_size:]
            
            # 1. Run a forward pass to get current prediction scores
            logits, _ = self.forward(idx_cond)
            
            # 2. Focus strictly on the very last token position in the sequence array
            logits = logits[:, -1, :] # Becomes shape: (B, C)
            
            # 3. Apply Softmax to convert raw scores into a valid probability distribution
            probs = F.softmax(logits, dim=-1) # Shape: (B, C)
            
            # 4. Sample a single character index randomly based on those distribution weights
            idx_next = torch.multinomial(probs, num_samples=1) # Shape: (B, 1)
            
            # 5. Append the newly generated token index to the running sequence matrix
            idx = torch.cat((idx, idx_next), dim=1) # Becomes shape: (B, T+1)
            
        return idx

# if __name__ == "__main__":
#     # Mock parameters representing our dataset shapes
#     vocab_size = 65
#     batch_size = 4
#     block_size = 8
    
#     # Initialize our model framework
#     model = NanoGPT(vocab_size=vocab_size, block_size=block_size)
    
#     # Generate fake integer batch matrices simulating input tokens and target labels
#     mock_inputs = torch.randint(0, vocab_size, (batch_size, block_size))
#     mock_targets = torch.randint(0, vocab_size, (batch_size, block_size))
    
#     # Execute a single forward test pass
#     logits, loss = model(mock_inputs, mock_targets)
    
#     print("--- STEP 8 ARCHITECTURE BASE VERIFICATION ---")
#     print(f"Input tensor batch shape:  {mock_inputs.shape}")
#     print(f"Output logits matrix shape: {logits.shape} (Flattened for loss calculation)")
#     print(f"Calculated Initial Loss:    {loss.item():.4f}")
#     # Note: For an untrained model guessing at random out of 65 options, 
#     # the baseline loss should sit near -ln(1/65) ≈ 4.17

# if __name__ == "__main__":
#     # Mock parameters for testing a single attention head
#     batch_size = 4
#     block_size = 8
#     n_embd = 128
#     head_size = 32
    
#     # Initialize one Single Attention Head
#     attention_head = Head(n_embd=n_embd, head_size=head_size, block_size=block_size)
    
#     # Create fake embedding inputs of shape (Batch, Time, Channels)
#     mock_embeddings = torch.randn(batch_size, block_size, n_embd)
    
#     # Pass the vectors through our Attention Head
#     output = attention_head(mock_embeddings)
    
#     print("--- STEP 9 ATTENTION HEAD SHAPE VERIFICATION ---")
#     print(f"Input Embedding Matrix Shape:  {mock_embeddings.shape}")
#     print(f"Attention Output Matrix Shape: {output.shape}")
#     # The output should successfully map down to your targeted head size dimension (32)!

if __name__ == "__main__":
    vocab_size = 65
    batch_size = 4
    block_size = 8
    
    # Initialize the complete model architecture stack
    model = NanoGPT(vocab_size=vocab_size, block_size=block_size)
    
    mock_inputs = torch.randint(0, vocab_size, (batch_size, block_size))
    mock_targets = torch.randint(0, vocab_size, (batch_size, block_size))
    
    logits, loss = model(mock_inputs, mock_targets)
    
    print("--- COMPLETE TRANSFORMER STACK VERIFICATION ---")
    print(f"Input batch shape:         {mock_inputs.shape}")
    print(f"Final output logits shape: {logits.shape}")
    print(f"Calculated random Loss:    {loss.item():.4f}")