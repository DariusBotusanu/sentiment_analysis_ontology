from pathlib import Path
import sys
import os

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
import torch

from data.data_handling import create_dataloaders, get_tokenizer
from tqdm import tqdm
from torch.nn import CrossEntropyLoss

NUM_EPOCHS = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = get_tokenizer()
train_dataloader, test_dataloader = create_dataloaders(tokenizer=tokenizer)


def train_bert_model(epochs=1):
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    optimizer = AdamW(model.parameters(), lr=1e-1, weight_decay=1e-2)

    loss_fn = CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch["labels"])

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

        # Check predictions per class
        with torch.no_grad():
            predictions = torch.argmax(outputs.logits, dim=-1)
            print(f"Epoch {epoch + 1} Class Distribution: {torch.bincount(predictions)}")

    return model

def save_model(model, tokenizer, save_dir="./fine_tuned_bert_financial_phrasebank"):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to {save_dir}")

def load_model_and_tokenizer(load_dir="./fine_tuned_bert_financial_phrasebank"):
    model = DistilBertForSequenceClassification.from_pretrained(load_dir)

    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(load_dir)

    print(f"Model and tokenizer loaded from {load_dir}")
    return model, tokenizer


def eval_model(model, model_name='BERT'):
    model.eval()
    predictions, true_labels = [], []

    # Collect predictions and true labels
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    cm_df = pd.DataFrame(
        cm,
        index=[f'Actual {i}' for i in range(len(cm))],
        columns=[f'Predicted {i}' for i in range(len(cm))]
    )

    # Save confusion matrix as CSV
    cm_df.to_csv(f'./{model_name}/confusion_matrix.csv')


if __name__=='__main__':
    model = train_bert_model(epochs=NUM_EPOCHS)
    save_model(model, tokenizer, save_dir = "./fine_tuned_bert_financial_phrasebank")
    model, tokenizer = load_model_and_tokenizer("./fine_tuned_bert_financial_phrasebank")
    eval_model(model, model_name="BERT")


