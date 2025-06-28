from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback


#labels, datacollector and early stopping needed 
def get_target_modules(model_name):
    if "progen" in model_name:
        return ["q_proj", "v_proj"]
    elif "rita" in model_name:
        return ["Wq", "Wv"]
    elif "protgpt2" in model_name:
        return ["c_attn"]
    return ["q_proj", "v_proj"]


def apply_lora(model, model_name, task_type=TaskType.CAUSAL_LM):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=get_target_modules(model_name),
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train_model(model, tokenizer, dataset, best_params, model_name, output_dir="./trained_model", use_lora=True):
    if use_lora:
        model = apply_lora(model, model_name)


    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        num_train_epochs=best_params["num_train_epochs"],
        weight_decay=best_params["weight_decay"],
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
        label_names=["labels"],
        metric_for_best_model="eval_loss",
        greater_is_better=False

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Model saved at {output_dir}")
