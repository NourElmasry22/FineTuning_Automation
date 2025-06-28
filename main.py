import argparse
import os
from model_loader import load_model_and_tokenizer
from preprocess import (
    read_fasta,
    PREPROCESSORS,
    MODEL_TO_PREPROCESSOR,
    split_dataset,
    save_splits
)
from datasets import load_dataset
from optuna_search import optuna_search
from trainer import train_model
from evaluate import evaluate_model
from generate import generate_sequence


def main():
    parser = argparse.ArgumentParser(description="Protein LLM Fine-Tuning Pipeline")

    parser.add_argument("--model", required=True, help="Model key (e.g., protgpt2, progen2-small, RITA_s)")
    parser.add_argument("--fasta_file", required=True, help="Path to input FASTA file")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of Optuna trials")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA PEFT")
    parser.add_argument("--generate", action="store_true", help="Generate sequences after training")
    parser.add_argument("--prompt", default="", help="Prompt for generation")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to generate")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

   
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model)

   
    print("Reading and preprocessing FASTA sequences...")
    raw_sequences = read_fasta(args.fasta_file)

    if args.model not in MODEL_TO_PREPROCESSOR:
        raise ValueError(f"Unknown model: {args.model}")

    preproc_key = MODEL_TO_PREPROCESSOR[args.model]
    processed_sequences = PREPROCESSORS[preproc_key](raw_sequences)


    print("Splitting dataset...")
    train, val, test = split_dataset(processed_sequences)
    save_splits(train, val, test, args.output_dir)

    print("Loading dataset...")
    dataset = load_dataset(
        "text",
        data_files={
            "train": f"{args.output_dir}/train.txt",
            "validation": f"{args.output_dir}/validation.txt",
            "test": f"{args.output_dir}/test.txt",
        }
    )


    print("Running Optuna hyperparameter search...")
    best_trial = optuna_search(
        model_init=lambda: load_model_and_tokenizer(args.model)[0],
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=f"{args.output_dir}/optuna",
        n_trials=args.n_trials
    )

    print("Best hyperparameters found:")
    print(best_trial)

    print("Training final model with best hyperparameters...")
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

 
    print("Evaluating model...")
    evaluate_model(
        model_dir=f"{args.output_dir}/final_model",
        test_file=f"{args.output_dir}/test.txt"
    )


    if args.generate:
        print(" Generating sequences...")
        for i in range(args.num_return_sequences):
            output = generate_sequence(
                model_name=args.model,
                model_dir=f"{args.output_dir}/final_model",
                prompt=args.prompt,
                save_path=f"{args.output_dir}/final_model/generated_{i+1}.txt"
            )
            print(f"[Generated #{i+1}]\n{output}\n")

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
