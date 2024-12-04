from datasets import load_dataset
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

import pickle
from sklearn.preprocessing import LabelEncoder

def save_label_encoder(label_encoder, save_path="./label_encoder.pkl"):
    with open(save_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"LabelEncoder saved to {save_path}")

def load_label_encoder(load_path="./label_encoder.pkl"):
    with open(load_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"LabelEncoder loaded from {load_path}")
    return label_encoder


def get_tokenized_datasets(dataset = None, tokenizer = None):
    if dataset is None:
        # Load Financial Phrasebank Dataset
        dataset = load_dataset("financial_phrasebank", "sentences_allagree")
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    if tokenizer is None:
        tokenizer = get_tokenizer()

    # Encode the labels
    label_encoder = LabelEncoder()
    dataset = dataset.map(lambda examples: {'label': label_encoder.fit_transform([examples['label']])[0]})

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    save_label_encoder(label_encoder, save_path="./label_encoder.pkl")

    return tokenized_datasets

def create_dataloaders(tokenizer, batch_size=16):
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            truncation=True,
            padding=True,
            max_length=128  # Adjust based on your model's max sequence length
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

if __name__ == '__main__':
    tokenizer = get_tokenizer()
    train_dataloader, test_dataloader = create_dataloaders(tokenizer=tokenizer)
    for batch in train_dataloader:
        print(batch)
    print("Running data_handling.py from data module")

    model = train_bert_model()
    save_model(model, tokenizer, save_dir="./fine_tuned_bert_financial_phrasebank")
    model, tokenizer = load_model_and_tokenizer("./fine_tuned_bert_financial_phrasebank")
    eval_model(model, model_name="BERT")

    # Example usage
    sentences = [
        "The company reported significant growth in Q3 earnings.",
        "Stocks declined sharply in morning trading.",
        "Investors remain optimistic about the market trends."
    ]

    results = infer_with_model(sentences)

    # Print results
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Predicted Class: {result['predicted_class']}")
        print("---")




