from model import TransformerModel
from utils import load_model_only, generate_text, load_vocab
import torch
from config import ntokens, emsize, nhid, nlayers, nhead, dropout

# Load the saved vocabulary
vocab = load_vocab('vocab.pkl')

# Initialize model with parameters from config
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)

# Load the saved model checkpoint
checkpoint_path = "transformer_checkpoint.pth"
epoch, loss = load_model_only(model, checkpoint_path)
print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")

# Ensure model is in evaluation mode and on the correct device
model.eval()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Use the model to generate text
prompt = "The battlefield"
generated_text = generate_text(model, vocab, prompt, predict_len=5)  # No need to pass emsize here
print("Generated text:")
print(generated_text)
