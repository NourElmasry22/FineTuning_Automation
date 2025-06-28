from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import optuna
import torch

def model_init_func(model_init):
    def init():
        return model_init()
    return init

def optuna_search(model_init, tokenizer, dataset, output_dir="./optuna_results", n_trials=10, direction="minimize"):
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        }

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
        fp16=torch.cuda.is_available(),
        report_to="none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model_init=model_init_func(model_init),
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        # compute_metrics=lambda eval_pred: {"perplexity": torch.exp(torch.tensor(eval_pred.loss))}
    )

    best_trial = trainer.hyperparameter_search(
        hp_space=optuna_hp_space,
        n_trials=n_trials,
        direction=direction,
    )

    return best_trial
