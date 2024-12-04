from pathlib import Path
import sys

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import pandas as pd
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch

from data.data_handling import create_dataloaders, get_tokenizer

from tqdm import tqdm

NUM_EPOCHS = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = get_tokenizer()
train_dataloader, test_dataloader = create_dataloaders(tokenizer=tokenizer)


def train_bert_model():
    # Load BERT model for classification
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-4)

    ## Scheduler
    #num_training_steps = len(train_dataloader) * NUM_EPOCHS
    #lr_scheduler = get_scheduler(
    #    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    #)

    model.to(device)

    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} complete! Loss: {loss.item()}")

    return model

def save_model(model, tokenizer, save_dir="./fine_tuned_bert_financial_phrasebank"):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to {save_dir}")

def load_model_and_tokenizer(load_dir="./fine_tuned_bert_financial_phrasebank"):
    model = BertForSequenceClassification.from_pretrained(load_dir)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(load_dir)

    print(f"Model and tokenizer loaded from {load_dir}")
    return model, tokenizer


def eval_model(model, model_name='BERT'):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())


    # Generate the classification report
    report = classification_report(
        true_labels,
        predictions,
        target_names={2:'positive', 1:'neutral', 0:'negative'},
        output_dict=True
    )

    # Convert to DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Save to CSV
    report_df.to_csv(f'./{model_name}/classification_report.csv')

    print(f"Evaluation report saved to ./{model_name}/classification_report.csv")


def infer_with_model(sentences, model_dir="./fine_tuned_bert_financial_phrasebank"):
    # Load the model and tokenizer
    loaded_model = BertForSequenceClassification.from_pretrained(model_dir)
    loaded_tokenizer = BertTokenizer.from_pretrained(model_dir)

    loaded_model.to(device)

    # Prepare predictions list
    predictions = []

    # Perform inference on each sentence
    loaded_model.eval()
    with torch.no_grad():
        for text in sentences:
            # Tokenize the sentence
            inputs = loaded_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(device)

            # Get model outputs
            outputs = loaded_model(**inputs)
            logits = outputs.logits

            # Get predicted class
            predicted_class = torch.argmax(logits, dim=-1).cpu().item()

            # Store prediction with original text
            predictions.append({
                'text': text,
                'predicted_class': predicted_class
            })

    return predictions


if __name__=='__main__':
    model = train_bert_model()
    save_model(model, tokenizer, save_dir = "./fine_tuned_bert_financial_phrasebank")
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


