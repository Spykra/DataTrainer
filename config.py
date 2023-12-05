# Hyperparameters

vocab_size = 28780  # Vocabulary size (unchanged)
ntokens = vocab_size
emsize = 512        # Increase embedding size to 512
nhid = 512          # Increase hidden size to 512
nlayers = 16        # Increase the number of layers to 4
nhead = 16          # Increase the number of heads to 6
dropout = 0.2       # Increase dropout to 0.5 for regularization
batch_size = 32     # Increase batch size to 32
sequence_length = 100  # Increase sequence length to 100
learning_rate = 0.001  # Decrease learning rate for more fine-grained updates
num_epochs = 300    # Keep the number of epochs as 300, adjust based on early stopping
