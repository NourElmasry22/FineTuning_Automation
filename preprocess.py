import random
from datasets import load_dataset
import os

def read_fasta(fasta_file):
    sequences = []
    with open(fasta_file, 'r') as file:
        sequence = ""
        for line in file:
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    return sequences

def preprocess_protgpt2(sequences):
    def split_sequence(seq, line_length=60):
        return '\n'.join(seq[i:i+line_length] for i in range(0, len(seq), line_length))
    return ["<|endoftext|>\n" + split_sequence(seq) for seq in sequences]

def preprocess_progen(sequences):
    return sequences

def preprocess_rita(sequences):
    return sequences  

PREPROCESSORS = {
    "protgpt2": preprocess_protgpt2,
    "progen": preprocess_progen,
    "rita": preprocess_rita,
}

MODEL_TO_PREPROCESSOR = {
    "protgpt2": "protgpt2",
    "progen2-small": "progen",
    "progen2-medium": "progen",
    "progen2-large": "progen",
    "progen2-xlarge": "progen",
    "RITA_s": "rita",
    "RITA_m": "rita",
    "RITA_l": "rita",
    "RITA_xl": "rita",
}

def split_dataset(sequences):
    random.shuffle(sequences)
    n = len(sequences)
    train = sequences[:int(0.9*n)]
    valid = sequences[int(0.9*n):int(0.95*n)]
    test = sequences[int(0.95*n):]
    return train, valid, test

def save_splits(train, valid, test, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/train.txt", 'w') as f: f.write("\n".join(train))
    with open(f"{out_dir}/validation.txt", 'w') as f: f.write("\n".join(valid))
    with open(f"{out_dir}/test.txt", 'w') as f: f.write("\n".join(test))

