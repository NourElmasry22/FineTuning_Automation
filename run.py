import argparse
import os
from model_loader import load_model_and_tokenizer
from preprocess import read_fasta, preprocessors, split_dataset, save_splits
from optuna_search import optuna_search
from trainer import train_model
from datasets import load_dataset

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name: protgpt2, progen, rita, etc.")
    parser.add_argument("--fasta_file", required=True, help="Path to FASTA file")
    parser.add_argument("--output_dir", default="./outputs", help="Where to save processed data and models")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--use_lora", action="store_true", help="Apply LoRA during training")
    args = parser.parse_args()

    print("[INFO] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    print("[INFO] Reading and preprocessing FASTA sequences...")
    raw_sequences = read_fasta(args.fasta_file)
    if args.model not in preprocessors:
        raise ValueError(f"Model {args.model} has no defined preprocessor.")
    preprocessor = preprocessors[args.model]
    processed_sequences = preprocessor(raw_sequences)

    print("[INFO] Splitting data into train/val/test...")
    train, valid, test = split_dataset(processed_sequences)
    save_splits(train, valid, test, args.output_dir)

    print("[INFO] Loading dataset...")
    dataset = load_dataset("text", data_files={
        "train": f"{args.output_dir}/train.txt",
        "validation": f"{args.output_dir}/validation.txt",
        "test": f"{args.output_dir}/test.txt",
    })

    print("[INFO] Starting hyperparameter tuning with Optuna...")
    best_trial = optuna_search(
        model_init=lambda: load_model_and_tokenizer(args.model)[0],
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=f"{args.output_dir}/optuna",
        n_trials=args.n_trials,
        direction="maximize"
    )

    print("[INFO] Best Hyperparameters:")
    print(best_trial)

    print("[INFO] Retraining final model with best hyperparameters...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    train_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        best_params=best_trial.hyperparameters,
        model_name=args.model,
        output_dir=f"{args.output_dir}/final_model",
        use_lora=args.use_lora
    )

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    run()
