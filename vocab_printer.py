import pickle

def load_vocab(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_vocabulary_to_file(vocab, output_file):
    with open(output_file, 'w') as file:
        for index, word in vocab.itos.items():
            file.write(f"Token: {index}, Word: '{word}'\n")

def run_save_vocabulary():
    vocab_file = 'vocab.pkl'
    output_file = 'vocabulary.txt'  # Name of the file where you want to save the vocabulary
    vocab = load_vocab(vocab_file)
    save_vocabulary_to_file(vocab, output_file)
    print(f"Vocabulary saved to {output_file}")

if __name__ == "__main__":
    run_save_vocabulary()
