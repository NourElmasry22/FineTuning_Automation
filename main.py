import argparse
import os
from model_loader import load_model_and_tokenizer
from preprocess import read_fasta, preprocessors, split_dataset, save_splits
from datasets import load_dataset
from optuna_search import optuna_search
from trainer import train_model
from evaluate import evaluate_model
from generate import generate_sequences

def main():
    parser = argparse.ArgumentParser(description="Protein Language Model Fine-Tuning Pipeline")

    parser.add_argument("--model", required=True, help="Model key (e.g., protgpt2, progen2-small, RITA_s)")
    parser.add_argument("--fasta_file", required=True, help="Path to input FASTA file")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--generate", action="store_true", help="Generate sequences after training")
    parser.add_argument("--num_return_sequences", type=int, default=5, help="Number of sequences to generate")
    args = parser.parse_args()

    print("[INFO] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model)


    print("[INFO] Reading and preprocessing FASTA sequences...")
    raw_sequences = read_fasta(args.fasta_file)
    if args.model not in preprocessors:
        raise ValueError(f"No preprocessor defined for model: {args.model}")
    processed_sequences = preprocessors[args.model](raw_sequences)

    
    print("[INFO] Splitting dataset...")
    train, val, test = split_dataset(processed_sequences)
    save_splits(train, val, test, args.output_dir)

    
    print("[INFO] Loading dataset...")
    dataset = load_dataset("text", data_files={
        "train": f"{args.output_dir}/train.txt",
        "validation": f"{args.output_dir}/validation.txt",
        "test": f"{args.output_dir}/test.txt",
    })

   
    print("[INFO] Running Optuna hyperparameter search...")
    best_trial = optuna_search(
        model_init=lambda: load_model_and_tokenizer(args.model)[0],
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=f"{args.output_dir}/optuna",
        n_trials=args.n_trials,
        direction="maximize"
    )

    print("[INFO] Best hyperparameters found:")
    print(best_trial)

    
    print("[INFO] Training final model...")
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

    print("[INFO] Evaluating model...")
    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset["test"],
        model_name=args.model,
        output_dir=f"{args.output_dir}/final_model"
    )

    
    if args.generate:
        print("[INFO] Generating sequences...")
        generate_sequences(
            model_name=args.model,
            model=model,
            tokenizer=tokenizer,
            num_return_sequences=args.num_return_sequences,
            output_path=f"{args.output_dir}/final_model/generated.txt"
        )

    print("All steps completed successfully.")

if __name__ == "__main__":
    main()
