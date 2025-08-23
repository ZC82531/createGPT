import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Union, Tuple


class TransformerBlock(nn.Module):
    # This is like ONE COMPLETE PROCESSING STATION in our text factory
    # Each station does two main jobs: 1) Let words talk to each other, 2) Let each word think alone
    def __init__(
            self, 
            num_heads: int,  # How many people analyze word relationships
            n_embed: int,    # How detailed each word's description is
            block_size: int  # Maximum sentence length we can handle
        ):
        super(TransformerBlock, self).__init__()
        hidden_dim = n_embed // num_heads  # Split the work evenly among people
        self.mhsa = MultiHeadSelfAttention(num_heads, hidden_dim, n_embed, block_size)
        self.feed_forward = FeedForward(n_embed)
        self.norm1 = nn.LayerNorm(n_embed)  # "Clean up" word descriptions before group analysis
        self.norm2 = nn.LayerNorm(n_embed)  # "Clean up" word descriptions before individual thinking

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Job 1: Let words interact and learn from each other + keep original understanding
        x = x + self.mhsa(self.norm1(x))
        # Job 2: Let each word think individually about what it learned + keep previous understanding  
        x = x + self.feed_forward(self.norm2(x))
        return x


class FeedForward(nn.Module):
    # This is like a "thinking room" where each word goes to process new information
    # Think of it like: "Let me think deeper about what I just learned from other words"
    def __init__(
            self, 
            n_embed: int,           # How detailed each word's description starts
            extend_width: int=4,    # How much "thinking space" to use (4x bigger)
            dropout: float=0.2      # How often to "forget" some thoughts (prevents overthinking)
        ):
        super(FeedForward, self).__init__()
        # This is the "thinking process":
        self.layer = nn.Sequential(
            nn.Linear(n_embed, extend_width*n_embed),  # "Let me think about this more deeply" (expand)
            nn.ReLU(),                                 # "Remove any negative/confused thoughts"
            nn.Linear(extend_width*n_embed, n_embed),  # "Summarize my conclusion" (compress back)
            nn.Dropout(dropout)                        # "Sometimes forget some details to stay flexible"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Send each word through the "thinking room" to process what it learned
        return self.layer(x)


class MultiHeadSelfAttention(nn.Module):
    # This is like having MULTIPLE PEOPLE (6 people) all reading the same sentence
    # Each person focuses on different things: grammar, meaning, style, etc.
    def __init__(
            self, 
            num_heads: int,     # How many different people are reading
            hidden_dim: int,    # How much each person can focus on at once
            n_embed: int,       # How detailed each word's description is
            block_size: int,    # Maximum sentence length
            dropout: float=0.2  # How often people "zone out"
        ):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        # Create multiple "readers" - each person has their own perspective
        self.heads = nn.ModuleList([SingleHead(hidden_dim, n_embed, block_size) for _ in range(self.num_heads)])
        # After all people give their opinions, combine them into one final understanding
        self.project = nn.Linear(n_embed, n_embed)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Let each person (head) analyze the sentence in their own way
        out = torch.cat([sh(x) for sh in self.heads], dim=-1)
        # Combine everyone's insights into one coherent understanding
        out = self.project(out)
        # Sometimes ignore some insights to stay flexible
        out = self.drop(out)
        return out


class SingleHead(nn.Module):
    # This is like ONE PERSON reading a sentence and deciding what to focus on
    def __init__(
            self, 
            hidden_dim: int,    # How many details this person can remember at once
            n_embed: int,       # How rich each word's description is
            block_size: int,    # Maximum sentence length this person can handle
            dropout: float=0.2  # How often this person "zones out" to stay flexible
        ):
        super(SingleHead, self).__init__()
        # Create three "thinking processes" for each word:
        self.key = nn.Linear(n_embed, hidden_dim, bias=False)      # "What can I offer to help?"
        self.query = nn.Linear(n_embed, hidden_dim, bias=False)    # "What am I looking for?"
        self.value = nn.Linear(n_embed, hidden_dim, bias=False)    # "What information do I actually have?"
        self.drop = nn.Dropout(dropout)
        # Create a "no cheating" rule: can't look at words that come later in the sentence
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # How many sentences, how many words per sentence, how rich each word is
        
        # Each word asks three questions:
        k = self.key(x)    # "What help can I offer to other words?"
        q = self.query(x)  # "What kind of help am I looking for?"
        
        # Figure out which words should pay attention to which other words
        # It's like each word saying "Hey, you seem relevant to what I need!"
        weights = q @ k.transpose(-2, -1) * C**(-0.5)  # Calculate compatibility scores
        
        # Apply the "no cheating" rule - can't look at future words
        masked_weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        
        # Convert scores to percentages: "I'll pay 60% attention to word 2, 30% to word 1, etc."
        masked_probs = F.softmax(masked_weights, dim=-1)
        masked_probs = self.drop(masked_probs)  # Sometimes "zone out" randomly
        
        # Get the actual information from each word
        v = self.value(x)  # "Here's what I actually know/mean"
        
        # Combine information based on attention: focus more on relevant words
        out = masked_probs @ v  # Weighted mixture of information
        return out


class GPT(nn.Module):
    # This is the COMPLETE TEXT PREDICTION FACTORY
    # It takes a sentence and predicts what letter/word should come next
    def __init__(
            self, 
            vocab_size: int,  # How many different letters/characters we know (like 65 for Shakespeare)
            block_size: int,  # Longest sentence we can read at once (like 256 characters)
            n_embed: int,     # How detailed we make each letter's "description card" (like 384 facts)
            num_heads: int,   # How many people analyze relationships in each processing station (like 6)
            n_layers: int     # How many processing stations to chain together (like 6)
        ):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        # Station 1: Convert each letter to a rich "description card"
        self.embedding = nn.Embedding(vocab_size, n_embed)
        # Station 2: Add "position tags" so we know which letter came first, second, etc.
        self.positional_embedding_table = nn.Embedding(block_size, n_embed)
        # Stations 3-8: Multiple processing stations where letters learn from each other
        self.blocks = nn.Sequential(
            *[TransformerBlock(num_heads, n_embed, block_size) for _ in range(n_layers)],
        )
        # Station 9: Final cleanup before making prediction
        self.norm = nn.LayerNorm(n_embed)        
        # Station 10: Convert final understanding back to "which letter comes next?" 
        self.fc = nn.Linear(n_embed, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape  # How many sentences we're processing, how many letters in each sentence
        
        # Step 1: Convert letter numbers to rich description cards
        token_embeddings = self.embedding(x) # Turn [5,8,12] into rich descriptions
        
        # Step 2: Add position information ("this is 1st letter, this is 2nd letter, etc.")
        positional_embedding = self.positional_embedding_table(torch.arange(T, device=x.device))
        
        # Step 3: Combine letter meanings with position information
        token_embeddings = token_embeddings + positional_embedding
        
        # Step 4: Send through all processing stations (letters learn from each other)
        blocks_out = self.blocks(token_embeddings)
        
        # Step 5: Final cleanup
        blocks_out = self.norm(blocks_out)
        
        # Step 6: Convert back to predictions: "Which of the 65 letters comes next?"
        logits = self.fc(blocks_out) # Get scores for each possible next letter
        
        # Step 7: Reshape for easier processing (flatten everything)
        logits = logits.reshape(B*T, self.vocab_size)
        return logits

    def generate(self, idx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        # This is how we CREATE NEW TEXT after training
        # Like playing "finish the sentence" but the computer does it
        
        t = idx.shape[1]  # How many letters we're starting with
        
        # Generate new letters one at a time
        for _ in range(max_tokens):
            # Don't let our sentence get too long (memory limit)
            idx_cond = idx[:, -self.block_size:]
            
            # Ask the factory: "Given what we have so far, what comes next?"
            logits = self.forward(idx_cond)
            
            # Reshape to understand which position we're predicting for
            logits = logits.reshape(1, t, -1)
            
            # Only care about the VERY LAST position (what comes after everything)
            logits = logits[:, -1, :]
            
            # Convert scores to probabilities: "30% chance it's 'e', 20% chance it's 'a', etc."
            probs = F.softmax(logits, dim=1)
            
            # Pick a letter randomly based on these probabilities (not always the most likely)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Add this new letter to our sentence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Keep track of sentence length
            if t < self.block_size:
                t += 1
        return idx


if __name__ == "__main__":
    # Test the model with some example parameters
    vocab_size = 65      # Number of unique characters (Shakespeare dataset has ~65 chars)
    block_size = 256     # Maximum sequence length
    n_embed = 384        # Embedding dimension
    num_heads = 6        # Number of attention heads
    n_layers = 6         # Number of transformer layers

    # Create the model
    model = GPT(vocab_size, block_size, n_embed, num_heads, n_layers)
    # Create a dummy input tensor (batch_size=1, sequence_length=256)
    inp = torch.ones((1,256), dtype=torch.long)
    # Run a forward pass to test the model
    out = model(inp)
    print(out.shape)  # Should be [256, 65] (flattened batch*seq, vocab_size)