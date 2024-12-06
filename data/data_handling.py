import pandas as pd
from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

def get_tokenizer():
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def create_dataloaders(tokenizer, batch_size=16):
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    # Balancing function
    from collections import Counter

    def balance_classes(dataset):
        label_counts = Counter(dataset["label"])
        min_count = min(label_counts.values())  # Find the minimum count across classes
        balanced_data = {key: [] for key in dataset.column_names}

        # Gather equal amounts of data for each label
        label_limits = {label: 0 for label in label_counts.keys()}
        for i, label in enumerate(dataset["label"]):
            if label_limits[label] < min_count:
                for key in dataset.column_names:
                    balanced_data[key].append(dataset[key][i])
                label_limits[label] += 1

        return balanced_data

    # Balance training dataset
    balanced_train_data = balance_classes(dataset["train"])
    dataset["train"] = Dataset.from_dict(balanced_train_data)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            truncation=True,
            padding=True,
            max_length=45
        )

    # Tokenize the datasets
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Set format for PyTorch
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create DataLoaders
    train_dataloader = DataLoader(
        tokenized_datasets['train'],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets['test'],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    return train_dataloader, test_dataloader





