# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")
    
    # Training hyperparameters:
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training. Default=16.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs. Default=3.")
    parser.add_argument("--val_split", type=float, default=10.0,
                        help="Percentage of data to use for validation (0.0 to 100.0). Default=10.0 (10%%).")
    parser.add_argument("--test_split", type=float, default=10.0,
                        help="Percentage of data to use for testing (0.0 to 100.0). Default=10.0 (10%%).")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramNet(nn.Module):
    """Helper module to convert one-hot to embeddings and pass through MLP"""
    def __init__(self, vocab_size, k, embed_size, mlp):
        super().__init__()
        self.vocab_size = vocab_size
        self.k = k
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.mlp = mlp
    
    def forward(self, x):
        # x is (batch, k*vocab_size) - flattened one-hot
        batch_size = x.shape[0]
        # Reshape to recover token indices from one-hot
        x_reshaped = x.view(batch_size, self.k, self.vocab_size)
        indices = torch.argmax(x_reshaped, dim=2)  # (batch, k)
        # Use embedding
        embedded = self.embedding(indices)  # (batch, k, embed_size)
        # Flatten for MLP input
        flat_embed = embedded.reshape(batch_size, -1)  # (batch, k*embed_size)
        # Pass through MLP
        return self.mlp(flat_embed)  # (batch, vocab_size)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # Build MLP layers
        layers = []
        
        # Input layer: k concatenated embeddings -> embed_size
        layers.append(nn.Linear(k * embed_size, embed_size))
        layers.append(nn.SiLU())
        
        # Inner layers
        # embed size is the length of the embedding vector
        for _ in range(num_inner_layers):
            layers.append(nn.Linear(embed_size, embed_size))
            layers.append(nn.SiLU())
        
        # Output layer: embed_size -> vocab_size
        layers.append(nn.Linear(embed_size, vocab_size))
        
        mlp = nn.Sequential(*layers)
        
        # Create net module that converts one-hot to embeddings then passes through MLP
        self.net = KGramNet(vocab_size, k, embed_size, mlp)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        # 🔹 1. Setup shapes
        seq_len, batch_size = tokens_seq.shape
        
        # If your input batch shape is:
        # (seq_len=5, batch=3)
        # Then:
        # number of timesteps = 5
        # number of sequences in batch = 3
        # outputs will store logits for each timestep.
        outputs = []

        # 🔹 2. Loop over sequence in chunks
        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            # This is only for memory-saving.
            # If chunk_size=1, you process one timestep at a time.
            # For now, think of it as:
            # for t in 0..seq_len-1:
            #     do computation
            
            block_outputs = []
            
            # 🔹 3. Loop over each timestep t
            for t in range(start, end):
                batch_logits = []
                # This means:
                # You are now predicting the next token at position t
                # But you must compute the prediction for every batch element at this timestep.

                # 🔹 4. Loop over batch elements b
                for b in range(batch_size):
                    # You process each sequence in the batch independently.

                    # 🔹 5. Extract the last K tokens before t
                    if t < self.k:
                        # Case B — t < K (beginning of sequence)
                        # Example: K=3 and t=1.
                        # You don't have enough past tokens → so you pad with zeros:
                        # context_ids = [0, x₀]
                        # This produces a fixed-length K context window.
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        # Case A — t ≥ K (normal case)
                        # Example: K=3 and t=5.
                        # You extract:
                        # tokens_seq[2], tokens_seq[3], tokens_seq[4]
                        # Mathematically: (x_{t-3}, x_{t-2}, x_{t-1})
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    # 🔹 6. Convert context tokens → one-hot vectors
                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size
                    )
                    # If vocab_size=50257 and K=3:
                    # context_oh shape: (3, 50257)
                    # Each row is a one-hot vector for one token
                    
                    # Flatten to (1, K*vocab_size) for MLP input
                    context_flat = context_oh.flatten().float().unsqueeze(0)
                    
                    # 🔹 7. Pass through network to get logits
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                
                # Collect all batch predictions for this timestep
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            # Concatenate all timesteps in this chunk
            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        # Concatenate all chunks to get final output
        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs


################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################

class FeedForward(nn.Module):
    """Feed-forward network with expansion factor"""
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        hidden_dim = d_model * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model)
        )
    
    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention with causal masking"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project to query, key, value
        q = self.q_proj(x)  # (batch, seq, d_model)
        k = self.k_proj(x)  # (batch, seq, d_model)
        v = self.v_proj(x)  # (batch, seq, d_model)
        
        # Reshape to multiple heads
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        
        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, n_heads, seq, seq)
        
        # Apply causal mask - only attend to past tokens
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq, seq)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # (batch, n_heads, seq, head_dim)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (batch, seq, d_model)
        
        # Final projection
        output = self.o_proj(context)  # (batch, seq, d_model)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization architecture"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Pre-normalization for attention
        self.norm1 = RMSNorm(d_model)
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads)
        # Pre-normalization for feed-forward
        self.norm2 = RMSNorm(d_model)
        # Feed-forward network
        self.ffn = FeedForward(d_model)
    
    def forward(self, x):
        # Pre-norm with residual connection for attention
        x = x + self.attention(self.norm1(x))
        # Pre-norm with residual connection for feed-forward
        x = x + self.ffn(self.norm2(x))
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Compute RMS: sqrt(mean(x^2) + eps)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        # Scale by learned weight
        return self.weight * (x / norm)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        
        # (a) Start with torch.nn.Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # (b) Create transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_blocks)
        ])
        
        # Final normalization
        self.norm = RMSNorm(d_model)
        
        # (c) Final unembedding layer to vocabulary size
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        """
        # Transpose to batch-first: (batch, seq_len)
        x = tokens_seq.transpose(0, 1)
        
        # Embedding: (batch, seq_len, d_model)
        x = self.embedding(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Transpose back to match expected output
        logits = logits.transpose(0, 1)  # (seq_len, batch, vocab_size)
        
        return logits


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    """
    Implements nucleus sampling (top-p sampling).
    
    Algorithm:
    1. Convert logits to probabilities via softmax
    2. Sort tokens by probability (descending): p(1) ≥ p(2) ≥ ... ≥ p(vocab_size)
    3. Find smallest k where: p(1) + ... + p(k-1) < p ≤ p(1) + ... + p(k)
    4. Sample from the filtered distribution p(1), ..., p(k) only
    
    How varying p affects generated text:
    - p → 0.0 (very small): Only the most likely token(s) are considered
      → More deterministic, repetitive, conservative text
      → Lower diversity, but more coherent
  
    - p = 0.5-0.9: Moderate nucleus size
      → Balanced creativity and coherence
      → Good for most creative writing tasks
  
    - p = 0.9-0.95 (typical): Large nucleus, excludes only very unlikely tokens
      → More diverse and creative text
      → Still maintains reasonable coherence
  
    - p → 1.0: Includes almost all tokens (approaches full softmax sampling)
      → Maximum diversity and creativity
      → May produce less coherent or more random text
      → Can include very unlikely tokens that might be nonsensical
    
    Args:
        logits: Tensor of shape (vocab_size,) - raw model outputs
        p: Nucleus probability threshold (0.0 to 1.0)
    
    Returns:
        sampled_token_id: Integer token ID sampled from the nucleus
    """
    # Step 1: Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Step 2: Sort probabilities in descending order
    # sorted_probs: [p(1), p(2), ..., p(vocab_size)] where p(1) ≥ p(2) ≥ ...
    # sorted_indices: [idx_1, idx_2, ...] - original token indices in sorted order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Step 3: Calculate cumulative probabilities
    # cumulative_probs[i] = p(1) + p(2) + ... + p(i)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find indices where cumulative probability > p
    # This marks tokens beyond the nucleus threshold
    sorted_indices_to_remove = cumulative_probs > p
    
    # Shift to include the first token that makes cumulative prob >= p
    # This ensures we include token k where: p(1)+...+p(k-1) < p ≤ p(1)+...+p(k)
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False  # Always keep at least one token
    
    # Step 4: Filter the distribution to keep only nucleus tokens
    filtered_indices = sorted_indices[~sorted_indices_to_remove]
    filtered_probs = sorted_probs[~sorted_indices_to_remove]
    
    # Renormalize probabilities so they sum to 1
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    # Sample from the filtered distribution
    sample_idx = torch.multinomial(filtered_probs, 1).item()
    
    # Return the corresponding original token ID
    return filtered_indices[sample_idx].item()


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


################################################################################
# 8. Training
################################################################################

def compute_validation_loss(model, val_loader, device):
    """
    Compute average loss on validation set.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation set
        device: Device to run on
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for batch_tokens in val_loader:
            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)
            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)
            total_loss += loss.item()
            total_steps += 1
    
    avg_val_loss = total_loss / total_steps if total_steps > 0 else 0.0
    return avg_val_loss


def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a",
                    val_loader=None,
                    test_loader=None):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Train Avg Loss: {avg_loss:.4f}")
        
        # Compute validation loss if validation set is provided
        if val_loader is not None:
            val_loss = compute_validation_loss(model, val_loader, device)
            print(f"[{model_name}] *** End of Epoch {epoch} *** Val Loss: {val_loss:.4f}")
    
    # Compute test loss at the end of training if test set is provided
    if test_loader is not None:
        test_loss = compute_validation_loss(model, test_loader, device)
        print(f"\n[{model_name}] ========== FINAL TEST LOSS ==========")
        print(f"[{model_name}] Test Loss: {test_loss:.4f}")
        print(f"[{model_name}] ======================================\n")


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = 1e-3

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    # Split dataset into train, validation, and test
    # Convert percentages to fractions
    val_split_pct = args.val_split
    test_split_pct = args.test_split
    dataset_size = len(combined_dataset)
    
    # Validate percentages
    if val_split_pct < 0 or val_split_pct > 100:
        raise ValueError(f"val_split must be between 0 and 100, got {val_split_pct}")
    if test_split_pct < 0 or test_split_pct > 100:
        raise ValueError(f"test_split must be between 0 and 100, got {test_split_pct}")
    if val_split_pct + test_split_pct > 100:
        raise ValueError(f"val_split + test_split cannot exceed 100%, got {val_split_pct}% + {test_split_pct}% = {val_split_pct + test_split_pct}%")
    
    # Calculate sizes for each split (convert percentage to fraction)
    val_size = int((val_split_pct / 100.0) * dataset_size)
    test_size = int((test_split_pct / 100.0) * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    # Ensure we have at least some training data
    if train_size <= 0:
        raise ValueError(f"Invalid splits: train_size={train_size}. Reduce val_split and/or test_split. "
                        f"Current: val_split={val_split_pct}%, test_split={test_split_pct}%")
    
    # Create splits list (only include non-zero splits)
    splits = [train_size]
    if val_size > 0:
        splits.append(val_size)
    if test_size > 0:
        splits.append(test_size)
    
    # Adjust last split to account for rounding
    if sum(splits) != dataset_size:
        splits[-1] += dataset_size - sum(splits)
    
    if len(splits) > 1:
        split_datasets = torch.utils.data.random_split(
            combined_dataset, splits,
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        train_dataset = split_datasets[0]
        val_dataset = split_datasets[1] if val_size > 0 else None
        test_dataset = split_datasets[2] if test_size > 0 else None
        
        print(f"Dataset split: {train_size} train ({train_size/dataset_size*100:.1f}%), "
              f"{val_size} validation ({val_split_pct:.1f}%), "
              f"{test_size} test ({test_split_pct:.1f}%)")
    else:
        train_dataset = combined_dataset
        val_dataset = None
        test_dataset = None
        print(f"No validation or test set. Using all {dataset_size} samples for training.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle validation set
            num_workers=0,
            collate_fn=seq_collate_fn
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle test set
            num_workers=0,
            collate_fn=seq_collate_fn
        )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    transformer = TransformerModel(
        vocab_size=vocab_size,
        d_model=embed_size,
        n_heads=2,
        n_blocks=4
    ).to(device)

    models = {
        "kgram_mlp_seq": kgram_model,
        "lstm_seq": lstm_model,
        "transformer": transformer,
    }


    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,  # <--- Pass the user-specified prompt here
            val_loader=val_loader,  # Pass validation loader
            test_loader=test_loader  # Pass test loader
        )

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()
