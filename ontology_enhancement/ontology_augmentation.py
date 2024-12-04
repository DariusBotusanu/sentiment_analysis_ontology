from pathlib import Path
import sys

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import pandas as pd
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
import torch

from data.data_handling import create_dataloaders, get_tokenizer

from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from owlready2 import get_ontology

# Extract key terms from the ontology
def extract_terms_from_ontology(ontology):
    terms = []
    for cls in ontology.classes():
        terms.append(cls.name.lower())  # Collect class names as terms
    return terms

def enhance_with_smo_terms(text, stock_market_terms):
    for term in stock_market_terms:
        if term in text.lower():
            text = text.replace(term, f"[SMO_{term.upper()}]")  # Add a prefix for ontology terms
    return text

class AugmentedDataLoader:
    def __init__(self, dataloader, ontology, stock_market_terms, tokenizer):
        """
        Augmented DataLoader to dynamically augment batches using the ontology.
        """
        self.dataloader = dataloader
        self.ontology = ontology
        self.stock_market_terms = stock_market_terms
        self.tokenizer = tokenizer

    def __iter__(self):
        """
        Iterate over the original dataloader and augment each batch.
        """
        for batch in self.dataloader:
            augmented_batch = self.augment_batch(batch)
            yield augmented_batch

    def augment_batch(self, batch):
        """
        Augment a single batch using ontology terms and synonyms.
        """
        augmented_inputs = []
        augmented_labels = []

        for text, label in zip(batch["text"], batch["labels"]):
            text = text.lower()  # Ensure consistency for term matching
            augmented_inputs.append(text)  # Original text
            augmented_labels.append(label)

            # Find terms in the text that match ontology terms
            terms = [term for term in self.stock_market_terms if term in text]
            for term in terms:
                synonyms = [syn.name.lower() for syn in self.ontology.search(is_a=self.ontology[term])]
                for synonym in synonyms:
                    augmented_text = text.replace(term, synonym)
                    augmented_inputs.append(augmented_text)  # Augmented text
                    augmented_labels.append(label)  # Same label for augmented text

        # Tokenize the augmented texts
        tokenized_inputs = self.tokenizer(
            augmented_inputs,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        # Return augmented batch
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": torch.tensor(augmented_labels)
        }

def create_augmented_dataloaders(tokenizer, ontology, batch_size=16):

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
    train_dataloader = AugmentedDataLoader(
    dataloader=train_dataloader,
    ontology=ontology,
    stock_market_terms=stock_market_terms,
    tokenizer=tokenizer
)
    test_dataloader = DataLoader(
        tokenized_datasets['test'],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    return train_dataloader, test_dataloader


def infer_with_smo(sentences, model_dir="./fine_tuned_bert_financial_phrasebank", ontology=None):
    loaded_model = BertForSequenceClassification.from_pretrained(model_dir)
    loaded_tokenizer = BertTokenizer.from_pretrained(model_dir)
    loaded_model.to(device)

    predictions = []
    loaded_model.eval()

    with torch.no_grad():
        for text in sentences:
            enhanced_text = enhance_with_smo_terms(text, stock_market_terms)
            inputs = loaded_tokenizer(
                enhanced_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(device)

            outputs = loaded_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).cpu().item()
            predictions.append({'text': text, 'predicted_class': predicted_class})

    return predictions

def eval_model_with_ontology(model, stock_market_terms, model_name='BERT_Ontology'):
    model.eval()
    predictions, true_labels = [], []
    ontology_predictions, ontology_true_labels = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            batch_labels = batch["labels"].cpu().numpy()

            predictions.extend(batch_predictions)
            true_labels.extend(batch_labels)

            # Check for ontology terms in the input text
            for text, pred, label in zip(batch["input_ids"], batch_predictions, batch_labels):
                text_decoded = tokenizer.decode(text, skip_special_tokens=True).lower()
                if any(term in text_decoded for term in stock_market_terms):
                    ontology_predictions.append(pred)
                    ontology_true_labels.append(label)

    # Generate overall classification report
    overall_report = classification_report(
        true_labels,
        predictions,
        target_names={2: 'positive', 1: 'neutral', 0: 'negative'},
        output_dict=True
    )
    overall_df = pd.DataFrame(overall_report).transpose()
    overall_df.to_csv(f'./{model_name}/overall_classification_report.csv')
    print(f"Overall evaluation saved to ./{model_name}/overall_classification_report.csv")

    # Generate ontology-specific classification report
    if ontology_predictions and ontology_true_labels:
        ontology_report = classification_report(
            ontology_true_labels,
            ontology_predictions,
            target_names={2: 'positive', 1: 'neutral', 0: 'negative'},
            output_dict=True
        )
        ontology_df = pd.DataFrame(ontology_report).transpose()
        ontology_df.to_csv(f'./{model_name}/ontology_classification_report.csv')
        print(f"Ontology-specific evaluation saved to ./{model_name}/ontology_classification_report.csv")
    else:
        print("No samples containing ontology terms were found in the evaluation set.")



NUM_EPOCHS = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from model_training.bert_training_eval import train_bert_model, save_model, load_model_and_tokenizer, eval_model
if __name__=='__main__':
    smo_path = "../ontologies/stock-market-ontology.owl"
    ontology = get_ontology(smo_path).load()
    stock_market_terms = extract_terms_from_ontology(ontology)
    tokenizer = get_tokenizer()
    train_dataloader, test_dataloader = create_augmented_dataloaders(tokenizer, ontology, batch_size=16)
    model = train_bert_model()
    save_model(model, tokenizer, save_dir="./fine_tuned_bert_financial_ontology")
    model, tokenizer = load_model_and_tokenizer("./fine_tuned_bert_financial_ontology")

    eval_model_with_ontology(model, stock_market_terms, model_name='BERT_Ontology')

    print(f"Extracted {len(stock_market_terms)} terms from SMO ontology.")


