from pathlib import Path
import sys

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from model_training.bert_training_eval import *
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from tqdm import tqdm

from owlready2 import get_ontology

# Extract key terms from the ontology
def extract_terms_from_ontology(ontology):
    terms = []
    for cls in ontology.classes():
        terms.append(cls.name.lower())  # Collect class names as terms
    return terms

from datasets import Dataset

def create_augmented_dataloaders(tokenizer, stock_market_terms, ontology, batch_size=16):
    from functools import partial

    # Load dataset and split
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    # Augmentation function (batched processing)
    def augment_with_ontology(dataset, stock_market_terms, ontology):
        augmented_sentences = []
        augmented_labels = []

        for text, label in tqdm(zip(dataset["sentence"], dataset["label"]), total=len(dataset["sentence"])):
            text = text.lower()  # Ensure consistency for term matching

            # Find terms in the text that match ontology terms
            terms_found = [term for term in stock_market_terms if term in text]

            if terms_found:
                for term in terms_found:
                    try:
                        # Try to get the ontology class, wrapped in a try-except
                        ontology_class = ontology[term]

                        # Use more robust synonym extraction
                        synonyms = []
                        # Add some fallback methods for synonym extraction
                        if hasattr(ontology_class, 'hasAlternativeLabel'):
                            synonyms.extend([syn.lower() for syn in ontology_class.hasAlternativeLabel])

                        # If no synonyms found, use a simple fallback
                        if not synonyms:
                            # Maybe use some basic transformation or keep the original
                            synonyms = [term + '_alt']

                        # Augment with synonyms
                        for synonym in synonyms:
                            augmented_text = text.replace(term, synonym)
                            augmented_sentences.append(augmented_text)
                            augmented_labels.append(label)

                    except KeyError:
                        # Skip if term not found in ontology
                        continue

        print(f"Original sentences: {len(dataset['sentence'])}")
        print(f"Augmented sentences: {len(augmented_sentences)}")

        return augmented_sentences, augmented_labels

    # Apply augmentation to the train dataset (with caching disabled)
    augmented_sentences, augmented_labels = augment_with_ontology(dataset['train'], stock_market_terms, ontology)

    with open("output.txt", "w") as file:
        for text, label in zip(augmented_sentences, augmented_labels):
            file.write(f"{text}\t{label}\n")

    new_data = {
        "sentence": augmented_sentences,
        "label": augmented_labels,
    }

    new_dataset = Dataset.from_dict(new_data)
    combined_sentences = list(dataset['train']['sentence']) + augmented_sentences
    combined_labels = list(dataset['train']['label']) + augmented_labels

    combined_dataset = Dataset.from_dict({
        'sentence': combined_sentences,
        'label': combined_labels
    })

    dataset['train'] = combined_dataset
    dataset.save_to_disk('./augmented_train')
    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            truncation=True,
            padding=True,
            max_length=45  # Adjust based on your model's max sequence length
        )

    # Tokenize the datasets
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Set format for PyTorch
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create DataLoaders

    test_dataloader = DataLoader(
        tokenized_datasets['test'],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    train_dataloader = DataLoader(
        tokenized_datasets['train'],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    return train_dataloader, test_dataloader


NUM_EPOCHS = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from model_training.bert_training_eval import train_bert_model, save_model, load_model_and_tokenizer, eval_model
if __name__=='__main__':
    smo_path = "../ontologies/stock-market-ontology.owl"
    ontology = get_ontology(smo_path).load()
    stock_market_terms = extract_terms_from_ontology(ontology)
    print(stock_market_terms)
    tokenizer = get_tokenizer()
    train_dataloader, test_dataloader = create_augmented_dataloaders(tokenizer, stock_market_terms, ontology, batch_size=16)
    model = train_bert_model(epochs=NUM_EPOCHS)
    save_model(model, tokenizer, save_dir="./fine_tuned_bert_financial_ontology")
    model, tokenizer = load_model_and_tokenizer("./fine_tuned_bert_financial_ontology")

    eval_model(model, model_name='BERT_Ontology')

    print(f"Extracted {len(stock_market_terms)} terms from SMO ontology.")


