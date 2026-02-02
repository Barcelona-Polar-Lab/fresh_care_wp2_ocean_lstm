import torch
import torch.nn as nn
import numpy as np

# Test GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a simple LSTM model
class TestLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 20, batch_first=True)
        self.fc = nn.Linear(20, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Create model and move to GPU
model = TestLSTM().to(device)
print(f"Model device: {next(model.parameters()).device}")

# Create sample data and move to GPU
batch_size = 32
seq_len = 10
input_size = 10

X = torch.randn(batch_size, seq_len, input_size).to(device)
y = torch.randn(batch_size, 1).to(device)

print(f"Input tensor device: {X.device}")
print(f"Target tensor device: {y.device}")

if torch.cuda.is_available():
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e6:.1f} MB")

# Forward pass
output = model(X)
print(f"Output device: {output.device}")

# Compute loss
criterion = nn.MSELoss()
loss = criterion(output, y)
print(f"Loss: {loss.item():.6f}")

if torch.cuda.is_available():
    print(f"GPU memory after forward pass: {torch.cuda.memory_allocated()/1e6:.1f} MB")

# Backward pass
loss.backward()

if torch.cuda.is_available():
    print(f"GPU memory after backward pass: {torch.cuda.memory_allocated()/1e6:.1f} MB")

print("GPU test completed successfully!")
