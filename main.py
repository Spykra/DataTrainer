import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerModel
from utils import Vocabulary, WarmupThenStepLR,get_rotary_frequencies, load_data, pad_collate_fn, save_checkpoint, load_checkpoint, plot_training_loss, save_vocab
from config import ntokens, emsize, nhid, nlayers, nhead, dropout, batch_size, learning_rate, num_epochs
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

torch.manual_seed(0)

# Load datasets
vocab = Vocabulary(min_freq=2)
train_dataset, val_dataset, test_dataset = load_data(vocab)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=pad_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_collate_fn)

# Initialize the model
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = WarmupThenStepLR(optimizer, warmup_steps=40, step_size=10, gamma=0.95)

# Load checkpoint if available
start_epoch, best_val_loss = load_checkpoint(model, optimizer)
if start_epoch == 0:
    best_val_loss = float('inf')

# Early stopping parameters
patience = 3  # Number of epochs to wait after last time validation loss improved
patience_counter = 0

training_losses, validation_losses = [], []

# Training and validation loop
for epoch in range(start_epoch, num_epochs):
    # Training
    model.train()
    total_loss = 0.
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} Training")
    for batch_idx, (data, targets) in progress_bar:
        optimizer.zero_grad()
        freqs = get_rotary_frequencies(data.size(1), emsize)
        output = model(data, freqs)
        loss = criterion(output.view(-1, ntokens), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'batch_loss': loss.item()})

    average_loss = total_loss / len(train_loader)
    training_losses.append(average_loss)
    scheduler.step()
    save_vocab(vocab, 'vocab.pkl')

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for data, targets in val_loader:
            freqs = get_rotary_frequencies(data.size(1), emsize)
            output = model(data, freqs)
            loss = criterion(output.view(-1, ntokens), targets.view(-1))
            total_val_loss += loss.item()

    average_val_loss = total_val_loss / len(val_loader)
    validation_losses.append(average_val_loss)
    perplexity = math.exp(average_val_loss)
    print(f"Epoch {epoch}, Validation Loss: {average_val_loss:.5f}, Perplexity: {perplexity:.2f}")

    # Early stopping and saving the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        save_checkpoint(model, optimizer, epoch, average_val_loss, "transformer_checkpoint.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter > patience:
            print("Early stopping triggered")
            break

# Plot training and validation loss
plot_training_loss(training_losses, validation_losses)

# Testing loop
model.eval()
total_test_loss = 0.0
with torch.no_grad():
    for data, targets in test_loader:
        freqs = get_rotary_frequencies(data.size(1), emsize)
        output = model(data, freqs)
        loss = criterion(output.view(-1, ntokens), targets.view(-1))
        total_test_loss += loss.item()

average_test_loss = total_test_loss / len(test_loader)
test_perplexity = math.exp(average_test_loss)
print(f"Test Loss: {average_test_loss:.2f}, Test Perplexity: {test_perplexity:.2f}")
