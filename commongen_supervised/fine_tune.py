from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments


# Load pretrained GPT-2 model and tokenizer
model_name = "gpt2-large"  # or "gpt2-medium", "gpt2-large", etc. depending on the model size
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare your custom dataset
train_file = "../dataset/commongen/train.json"  # Path to your custom dataset file

if __name__ == '__main__':

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_file,
        tokenizer=tokenizer,
    )

    # Start fine-tuning
    trainer.train()
