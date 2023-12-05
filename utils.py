import os
import pickle
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import string
from torch.utils.data import Dataset
from config import sequence_length, emsize
from datasets import load_dataset


# Custom Scheduler with Warm-up
class WarmupThenStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, step_size, gamma=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        super(WarmupThenStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            return [base_lr * self._step_count / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # After warmup_steps, use StepLR logic
            return [base_lr * self.gamma ** ((self._step_count - self.warmup_steps) // self.step_size) for base_lr in self.base_lrs]


class Vocabulary:
    def __init__(self, min_freq=1):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.min_freq = min_freq

    def build_vocabulary(self, sentence_list):
        counter = Counter()
        for sentence in sentence_list:
            counter.update(self.tokenize(sentence))

        freqs = dict(filter(lambda x: x[1] >= self.min_freq, counter.items()))
        for word in freqs:
            self.itos[len(self.itos)] = word
            self.stoi[word] = len(self.stoi)

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

    @staticmethod
    def tokenize(text):
        text = text.lower()
        text = ''.join([ch for ch in text if ch not in string.punctuation])
        return text.split()

def pad_collate_fn(batch):
    (xx, yy) = zip(*batch)
    xx_pad = pad_sequence([torch.tensor(x) for x in xx], padding_value=0, batch_first=True)
    yy_pad = pad_sequence([torch.tensor(y) for y in yy], padding_value=0, batch_first=True)

    return xx_pad, yy_pad


class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(vocab):
    dataset = load_dataset('wikitext', 'wikitext-2-v1')

    # Flatten the list of texts into a single list of sentences for the training data
    train_sentences = sum([text.split('\n') for text in dataset['train']['text']], [])

    # Build the vocabulary with sentences from the train set
    vocab.build_vocabulary(train_sentences)

    train_data = preprocess_data(dataset['train']['text'], vocab, sequence_length)
    val_data = preprocess_data(dataset['validation']['text'], vocab, sequence_length)
    test_data = preprocess_data(dataset['test']['text'], vocab, sequence_length)

    return TextDataset(train_data, vocab), TextDataset(val_data, vocab), TextDataset(test_data, vocab)




def preprocess_data(texts, vocab, sequence_length):
    tokenized_data = []

    for text in texts:
        numericalized_text = vocab.numericalize(text)
        for i in range(0, len(numericalized_text) - sequence_length, sequence_length):
            input_sequence = numericalized_text[i:i + sequence_length]
            target_sequence = numericalized_text[i + 1:i + sequence_length + 1]
            tokenized_data.append((input_sequence, target_sequence))

    return tokenized_data


def plot_training_loss(training_losses, validation_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def save_checkpoint(model, optimizer, epoch, loss, filename):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename="transformer_checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    return 0, float('inf')


def generate_text(model, vocab, initial_text, predict_len):
    model.eval()
    
    # Tokenize and numericalize the initial text
    numericalized = [vocab.stoi[token] if token in vocab.stoi else vocab.stoi["<UNK>"] for token in vocab.tokenize(initial_text)]
    input_sequence = torch.tensor([numericalized], dtype=torch.long).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        for _ in range(predict_len):
            # Generate frequency tensor for current input_sequence length
            seq_len = input_sequence.size(1)
            freqs = get_rotary_frequencies(seq_len, model.ninp)  # model.ninp is the embedding size

            # Pass frequency tensor to model along with the input_sequence
            output = model(input_sequence, freqs)
            last_token_logits = output[0, -1, :]
            predicted_token_id = torch.argmax(last_token_logits).item()
            numericalized.append(predicted_token_id)

            # Update input_sequence for next iteration
            input_sequence = torch.tensor([numericalized], dtype=torch.long).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Convert numericalized tokens back to text
    return ' '.join([vocab.itos[token_id] for token_id in numericalized])




def load_model_only(model, filename="transformer_checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Not available')
        loss = checkpoint.get('loss', 'Not available')
        return epoch, loss
    else:
        print("No checkpoint found at:", filename)
        return None, None
    

def save_vocab(vocab, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_rotary_frequencies(seq_len, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
    t = torch.arange(seq_len)
    freqs = torch.ger(t, inv_freq)

    # Combine sin and cos terms
    freqs = torch.stack((freqs.sin(), freqs.cos()), dim=-1)
    freqs = freqs.flatten(start_dim=1)

    # Ensure freqs has the same feature_dim as expected
    if freqs.shape[1] != dim:
        freqs = torch.cat([freqs, freqs], dim=1)  # Duplicate to match the dimension

    return freqs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

