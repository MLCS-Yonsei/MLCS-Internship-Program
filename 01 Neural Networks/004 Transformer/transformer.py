#==========================================#
# Title:  Machine Translation with Transformer
# Author: Dongjae Kim
# Date:   2025-01-05
# Reference: Vaswani et al. "Attention is all you need" (2017)
#==========================================#
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# Hyperparameters
# ============================================================================
# Option 1: Original settings (from "Attention is all you need" paper)
# Warning: Causes overfitting on this simple toy task!
# ============================================================================
# d_model = 512
# nhead = 8
# num_encoder_layers = 6
# num_decoder_layers = 6
# dim_feedforward = 2048
# dropout = 0.1
# max_seq_length = 100
# batch_size = 32
# epochs = 200
# learning_rate = 0.0001
# weight_decay = 0.0

# ============================================================================
# Option 2: Adjusted settings to prevent overfitting 
# ============================================================================
d_model = 256          # Embedding dimension (reduced from 512)
nhead = 8              # Number of attention heads
num_encoder_layers = 3 # Number of encoder layers (reduced from 6)
num_decoder_layers = 3 # Number of decoder layers (reduced from 6)
dim_feedforward = 1024 # Dimension of feedforward network (reduced from 2048)
dropout = 0.3          # Increased dropout for regularization (from 0.1)
max_seq_length = 100
batch_size = 32
epochs = 200           # Adjusted epochs (from 70)
learning_rate = 0.0005 # Slightly increased learning rate (from 0.0001)
weight_decay = 0.0001  # L2 regularization (added)

"""
Step 1: Positional Encoding
The Transformer doesn't have recurrence or convolution, so we need to inject 
information about the position of tokens in the sequence.
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model) for positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the positional encodings using sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

"""
Step 2: Define the Transformer Model
We'll use PyTorch's built-in Transformer module for simplicity,
but you can also implement it from scratch using MultiheadAttention.
"""
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, 
                 num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None):
        # src shape: (batch_size, src_seq_len)
        # tgt shape: (batch_size, tgt_seq_len)
        
        # Embedding and positional encoding
        src_emb = self.dropout(self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        
        # Transformer forward pass
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Output projection
        output = self.fc_out(output)
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a mask to prevent attention to future positions."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

"""
Step 3: Create a Simple Toy Dataset
For demonstration, we'll create a simple sequence-to-sequence task:
reversing sequences of numbers.
"""
class ReverseSequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=10, vocab_size=20):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Special tokens
        self.PAD_IDX = 0
        self.SOS_IDX = 1  # Start of sequence
        self.EOS_IDX = 2  # End of sequence
        
        self.data = []
        for _ in range(num_samples):
            # Generate random sequence (excluding special tokens)
            seq = np.random.randint(3, vocab_size, size=seq_length)
            # Target is the reversed sequence
            target = seq[::-1].copy()
            
            # Add SOS and EOS tokens
            src = np.concatenate([[self.SOS_IDX], seq, [self.EOS_IDX]])
            tgt = np.concatenate([[self.SOS_IDX], target, [self.EOS_IDX]])
            
            self.data.append((src, tgt))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0], dtype=torch.long), \
               torch.tensor(self.data[idx][1], dtype=torch.long)

"""
Step 4: Prepare Data Loaders
"""
vocab_size = 20  # Small vocabulary for toy task
train_dataset = ReverseSequenceDataset(num_samples=5000, seq_length=8, vocab_size=vocab_size)
test_dataset = ReverseSequenceDataset(num_samples=500, seq_length=8, vocab_size=vocab_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Check data shape
src_sample, tgt_sample = next(iter(train_loader))
print(f"Source shape: {src_sample.shape}")  # (batch_size, seq_len)
print(f"Target shape: {tgt_sample.shape}")  # (batch_size, seq_len)
print(f"Sample source sequence: {src_sample[0]}")
print(f"Sample target sequence: {tgt_sample[0]}")

"""
Step 5: Initialize Model, Loss, and Optimizer
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = TransformerModel(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.PAD_IDX, label_smoothing=0.1)  # Label smoothing for regularization
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)

# Learning rate scheduler with warmup and decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

"""
Step 6: Training Loop

Anti-overfitting measures implemented:
1. Reduced model capacity (smaller d_model, fewer layers)
2. Increased dropout (0.3 instead of 0.1)
3. Label smoothing (0.1) to prevent overconfidence
4. L2 regularization (weight_decay)
5. ReduceLROnPlateau scheduler (adapts learning rate based on test loss)
6. Early stopping (stops if test loss increases significantly)
"""
train_losses = []
test_losses = []

print("\n" + "="*50)
print("Starting Training...")
print("="*50)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src = src.to(device)  # (batch_size, src_seq_len)
        tgt = tgt.to(device)  # (batch_size, tgt_seq_len)
        
        # Prepare input and target for teacher forcing
        tgt_input = tgt[:, :-1]  # Remove last token for input
        tgt_output = tgt[:, 1:]  # Remove first token (SOS) for target
        
        # Create target mask (causal mask)
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
        # Forward pass
        output = model(src, tgt_input, tgt_mask=tgt_mask)
        
        # Reshape for loss calculation
        output = output.reshape(-1, vocab_size)
        tgt_output = tgt_output.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, tgt_output)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f'Epoch [{str(epoch+1).zfill(2)}/{epochs}], '
                  f'Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            output = output.reshape(-1, vocab_size)
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
            total_test_loss += loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    print(f'Epoch [{str(epoch+1).zfill(2)}/{epochs}] - '
          f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    
    # Update learning rate based on test loss
    scheduler.step(avg_test_loss)
    
    # Early stopping if overfitting gets too bad
    if len(test_losses) > 10 and avg_test_loss > min(test_losses[-10:]) * 1.5:
        print(f"\nEarly stopping at epoch {epoch+1} due to overfitting")
        break

"""
Step 7: Evaluation and Visualization
"""
print("\n" + "="*50)
print("Evaluating Model...")
print("="*50)

def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    """Simple greedy decoding for inference."""
    src = src.to(device)
    model.eval()
    
    with torch.no_grad():
        # Encode the source
        src_emb = model.pos_encoder(model.src_embedding(src) * math.sqrt(model.d_model))
        memory = model.transformer.encoder(src_emb)
        
        # Start with SOS token
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        
        for i in range(max_len - 1):
            tgt_emb = model.pos_encoder(model.tgt_embedding(ys) * math.sqrt(model.d_model))
            tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).to(device)
            
            out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            out = model.fc_out(out)
            
            # Get the last token prediction
            prob = out[:, -1, :]
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            
            # Append predicted token
            ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(device)], dim=1)
            
            # Stop if EOS token is predicted
            if next_word == end_symbol:
                break
    
    return ys

# Test on some examples
num_examples = 5
model.eval()

print("\nSample Predictions:")
print("-" * 50)

for i in range(num_examples):
    src, tgt = test_dataset[i]
    src_seq = src[1:-1].tolist()  # Remove SOS and EOS
    tgt_seq = tgt[1:-1].tolist()  # Remove SOS and EOS
    
    # Predict
    predicted = greedy_decode(
        model, 
        src.unsqueeze(0), 
        max_len=len(src), 
        start_symbol=train_dataset.SOS_IDX,
        end_symbol=train_dataset.EOS_IDX
    )
    predicted_seq = predicted.squeeze().tolist()
    # Handle both list and single value cases
    if isinstance(predicted_seq, int):
        predicted_seq = [predicted_seq]
    # Remove SOS and EOS tokens
    predicted_seq = [token for token in predicted_seq if token not in [train_dataset.SOS_IDX, train_dataset.EOS_IDX]]
    
    print(f"\nExample {i+1}:")
    print(f"  Input:     {src_seq}")
    print(f"  Target:    {tgt_seq}")
    print(f"  Predicted: {predicted_seq}")
    print(f"  Correct:   {predicted_seq == tgt_seq}")

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Transformer Training Progress')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'training_curve.png'), dpi=300) # save the figure
plt.show()

print("\n" + "="*50)
print("Training Complete!")
print("="*50)

"""
Output example:
==================================================
Evaluating Model...
==================================================

Sample Predictions:
--------------------------------------------------

Example 1:
  Input:     [18, 10, 10, 7, 17, 4, 14, 17]
  Target:    [17, 14, 4, 17, 7, 10, 10, 18]
  Predicted: [17, 14, 4, 17, 7, 10, 10, 18]
  Correct:   True

Example 2:
  Input:     [15, 9, 19, 10, 11, 16, 8, 6]
  Target:    [6, 8, 16, 11, 10, 19, 9, 15]
  Predicted: [6, 8, 16, 11, 10, 19, 9, 15]
  Correct:   True

Example 3:
  Input:     [15, 11, 16, 10, 7, 19, 11, 14]
  Target:    [14, 11, 19, 7, 10, 16, 11, 15]
  Predicted: [14, 11, 19, 7, 10, 16, 11, 15]
  Correct:   True

Example 4:
  Input:     [4, 12, 7, 13, 5, 9, 9, 18]
  Target:    [18, 9, 9, 5, 13, 7, 12, 4]
  Predicted: [18, 9, 9, 5, 13, 7, 12, 4]
  Correct:   True

Example 5:
  Input:     [6, 11, 19, 19, 17, 10, 18, 14]
  Target:    [14, 18, 10, 17, 19, 19, 11, 6]
  Predicted: [14, 18, 10, 17, 19, 19, 11, 6]
  Correct:   True

==================================================
Training Complete!
==================================================
"""