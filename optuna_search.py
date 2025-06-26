from transformers import Trainer, TrainingArguments
import optuna
import torch

def model_init_func(model_init):
    def init():
        return model_init()
    return init

def compute_metrics(eval_pred):
    return {}  

def optuna_search(
    model_init,
    tokenizer,
    dataset,
    output_dir="./optuna_results",
    n_trials=10,
    direction="maximize",
):
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model_init=model_init_func(model_init),
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    best_trial = trainer.hyperparameter_search(
        hp_space=optuna_hp_space,
        n_trials=n_trials,
        direction=direction,
    )

    return best_trial
