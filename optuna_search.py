from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import optuna
import torch
from transformers import EarlyStoppingCallback
def model_init_func(model_init):
    def init():
        return model_init()
    return init

def optuna_search(model_init, tokenizer, dataset, output_dir="./optuna_results", n_trials=10, direction="minimize"):
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True), 
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 10),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        }

   
    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)


    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
        fp16=False,
        report_to="none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        max_grad_norm=1.0, 
        logging_steps=50,
        save_steps=1000,
        eval_steps=1000,
        gradient_checkpointing=True,  
        dataloader_drop_last=True,  
        ignore_data_skip=True,
    
    )

    trainer = Trainer(
        model_init=model_init_func(model_init),
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        # compute_metrics=lambda eval_pred: {"perplexity": torch.exp(torch.tensor(eval_pred.loss))}
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    best_trial = trainer.hyperparameter_search(
        hp_space=optuna_hp_space,
        n_trials=n_trials,
        direction=direction,
    )

     # If no trials completed successfully, return default parameters
    if best_trial is None:
        print("[WARNING] No trials completed successfully, using default parameters")
        class DefaultTrial:
            def __init__(self):
                self.hyperparameters = {
                    "learning_rate": 5e-5,
                    "per_device_train_batch_size": 4,
                    "num_train_epochs": 2,
                    "weight_decay": 0.01,
                }
        return DefaultTrial()

    return best_trial
