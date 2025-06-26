import argparse
from model_loader import load_model_and_tokenizer
from preprocess import read_fasta, preprocessors, split_dataset, save_splits
from optuna_search import optuna_search
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name: protgpt2, progen, rita, etc.")
    parser.add_argument("--fasta_file", required=True, help="Path to FASTA file")
    parser.add_argument("--output_dir", default="./outputs", help="Where to save processed data and models")
    args = parser.parse_args()


    model, tokenizer = load_model_and_tokenizer(args.model)


    raw_sequences = read_fasta(args.fasta_file)
    if args.model not in preprocessors:
        raise ValueError(f"Model {args.model} has no defined preprocessor.")
    preprocessor = preprocessors[args.model]
    processed_sequences = preprocessor(raw_sequences)

    
    train, valid, test = split_dataset(processed_sequences)
    save_splits(train, valid, test, args.output_dir)

    
    dataset = load_dataset("text", data_files={
        "train": f"{args.output_dir}/train.txt",
        "validation": f"{args.output_dir}/validation.txt",
        "test": f"{args.output_dir}/test.txt",
    })

    
    best_trial = optuna_search(
        model_init=lambda: load_model_and_tokenizer(args.model)[0],  
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=f"{args.output_dir}/optuna",
        n_trials=10,
        direction="maximize"
    )

    print(" Best Hyperparameters:")
    print(best_trial)

if __name__ == "__main__":
    main()
