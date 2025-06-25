# EthioMart/src/model_finetuner.py

import pandas as pd
import re
from pathlib import Path
import logging
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset, Features, Value, ClassLabel, Sequence
import torch
from collections import Counter

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

import sys

# Add the project root to sys.path to allow importing from src and config
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import configurations from config/config.py
try:
    from config.config import DATA_DIR
except ImportError:
    logging.error("Could not import DATA_DIR from config.config. Please ensure the config file is correct.")
    DATA_DIR = Path(__file__).parent.parent / "data"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_CHECKPOINT = "Davlan/afro-xlmr-large"
OUTPUT_DIR = Path(project_root / "models" / "afro_xlmr_ner_fine_tuned") 
CONLL_FILE_PATH = Path(DATA_DIR.parent / "labeled" / "telegram_ner_data_rule_based.conll")

# Training arguments (can be adjusted)
TRAINING_ARGS = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "training_logs"),
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_dir=str(OUTPUT_DIR / "runs"),
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="tensorboard",
)

# --- Data Loading and Parsing ---
def read_conll_file(file_path):
    """
    Reads a CoNLL-formatted file and parses it into a list of dictionaries,
    where each dictionary represents a sentence with tokens and NER tags.
    """
    texts = []
    tokens = []
    ner_tags = []
    
    all_possible_labels = ["O", 
                           "B-PRODUCT", "I-PRODUCT", "L-PRODUCT", "U-PRODUCT",
                           "B-PRICE", "I-PRICE", "L-PRICE", "U-PRICE",
                           "B-LOC", "I-LOC", "L-LOC", "U-LOC",
                           "B-CONTACT", "I-CONTACT", "L-CONTACT", "U-CONTACT",
                           "B-DELIVERY", "I-DELIVERY", "L-DELIVERY", "U-DELIVERY"]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    tokens.append(parts[0])
                    tag = parts[1]
                    if tag not in all_possible_labels:
                        logging.warning(f"Unknown tag '{tag}' found for token '{parts[0]}'. Changing to 'O'.")
                        ner_tags.append("O")
                    else:
                        ner_tags.append(tag)
                else:
                    logging.warning(f"Skipping malformed line (too many/few parts): '{line}'")
            else:
                if tokens:
                    texts.append({"tokens": tokens, "ner_tags": ner_tags})
                    tokens = []
                    ner_tags = []
        if tokens:
            texts.append({"tokens": tokens, "ner_tags": ner_tags})
            
    logging.info(f"Loaded {len(texts)} sentences from CoNLL file.")
    
    all_flat_tags = [tag for sent in texts for tag in sent['ner_tags']]
    logging.info(f"Class distribution: {Counter(all_flat_tags)}")

    return texts, all_possible_labels

# --- Tokenization and Label Alignment ---
def tokenize_and_align_labels(examples, tokenizer, label_to_id, id_to_label):
    """
    Tokenizes a batch of token lists and aligns the NER labels with the new subword tokens.
    Handles 'O' (Outside), 'B-' (Beginning), 'I-' (Inside), 'L-' (Last), 'U-' (Unit-length/Single) tags.
    
    This function processes `examples["tokens"]` which is expected to be a list of lists of strings,
    e.g., `[["token1", "token2"], ["tokenA", "tokenB"]]`.
    """
    # Ensure examples["tokens"] is a list of lists of strings, even if some are empty.
    # The `is_split_into_words=True` expects this format.
    # Pass `None` for empty lists of tokens, as tokenizer can't process them
    # and this often indicates an issue with data input.
    # Instead, we will handle empty lists gracefully within the loop below.

    # Tokenize the batch of sentences
    # The `examples["tokens"]` is already a list of lists of strings (sentences).
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    # Iterate through each example in the batch
    for i, word_label_ids in enumerate(examples["ner_tags"]): # word_label_ids will be a list of integers for current sentence
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        # Guard against empty `word_ids` if a sentence was too short or tokenized to nothing
        if not word_ids:
            labels.append([-100]) # Append a placeholder so `labels` list has correct length
            continue

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Special tokens
            elif word_idx != previous_word_idx:
                # This is the first token of a new word. Get original string label.
                # Ensure word_idx is within bounds of word_label_ids
                if word_idx < len(word_label_ids):
                    original_label_id = word_label_ids[word_idx]
                    original_label_str = id_to_label[original_label_id] # Convert ID to string
                    
                    if original_label_str.startswith("B-"):
                        label_ids.append(original_label_id)
                    elif original_label_str.startswith("I-") or \
                         original_label_str.startswith("L-") or \
                         original_label_str.startswith("U-"):
                        entity_type = original_label_str.split("-")[1]
                        label_ids.append(label_to_id["B-" + entity_type])
                    else: # 'O' tag
                        label_ids.append(original_label_id)
                else:
                    # This implies a mismatch. Assign -100 to be safe.
                    logging.warning(f"Word index {word_idx} out of bounds for word labels. Assigning -100.")
                    label_ids.append(-100)
            else:
                # This is a subsequent token of a word that has been split into subwords
                if word_idx < len(word_label_ids):
                    original_label_id = word_label_ids[word_idx]
                    original_label_str = id_to_label[original_label_id]
                    
                    if original_label_str.startswith("B-") or \
                       original_label_str.startswith("U-"):
                        entity_type = original_label_str.split("-")[1]
                        label_ids.append(label_to_id["I-" + entity_type])
                    elif original_label_str.startswith("I-") or \
                         original_label_str.startswith("L-"):
                        label_ids.append(original_label_id) # Keep original I- or L- if it's not the start
                    else: # 'O' tag
                        label_ids.append(original_label_id)
                else:
                    # Mismatch. Assign -100.
                    logging.warning(f"Word index {word_idx} out of bounds for word labels (subsequent token). Assigning -100.")
                    label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# --- Evaluation Metrics ---
def compute_metrics(p, label_list):
    """
    Computes and returns precision, recall, and f1-score for NER.
    """
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=2)

    # Remove ignored index (where label is -100)
    true_predictions = [
        [label_list[p_id] for (p_id, l_id) in zip(prediction, label) if l_id != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l_id] for (p_id, l_id) in zip(prediction, label) if l_id != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    
    logging.info(f"\n--- Classification Report ---\n{classification_report(true_labels, true_predictions)}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    # 1. Load the labeled dataset
    if not CONLL_FILE_PATH.exists():
        logging.error(f"❌ CoNLL file not found at {CONLL_FILE_PATH}. Please ensure data_labeler.py has run and produced this file.")
        sys.exit(1)
    
    raw_data, all_possible_labels = read_conll_file(CONLL_FILE_PATH)

    if not raw_data:
        logging.error("❌ No data loaded from CoNLL file. Exiting.")
        sys.exit(1)

    # 2. Split the dataset into train, validation, and test sets
    
    # Create a simpler stratification key: based on presence of entities to reduce unique classes
    stratify_keys = []
    for sentence in raw_data:
        has_entity = any(tag != 'O' for tag in sentence['ner_tags'])
        stratify_keys.append("HAS_ENTITY" if has_entity else "NO_ENTITY")

    unique_stratify_keys_counts = Counter(stratify_keys)
    viable_stratify = True
    if len(unique_stratify_keys_counts) > 1:
        for key, count in unique_stratify_keys_counts.items():
            if count < 2:
                viable_stratify = False
                logging.warning(f"Stratification key '{key}' has only {count} member(s). Disabling stratification for robustness.")
                break
    else:
        viable_stratify = False
        logging.info("Only one unique stratification key found or not enough samples per class. Performing non-stratified split.")
    
    # Perform splits
    all_indices = list(range(len(raw_data)))
    if viable_stratify:
        logging.info("Performing stratified split.")
        train_val_indices, test_indices = train_test_split(
            all_indices, test_size=0.1, random_state=42, stratify=stratify_keys
        )
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.111, random_state=42, stratify=[stratify_keys[i] for i in train_val_indices]
        )
    else:
        logging.info("Performing non-stratified split.")
        train_val_indices, test_indices = train_test_split(all_indices, test_size=0.1, random_state=42)
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.111, random_state=42)

    train_data = [raw_data[i] for i in train_indices]
    val_data = [raw_data[i] for i in val_indices]
    test_data = [raw_data[i] for i in test_indices]

    logging.info(f"Dataset split: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

    # 3. Create Hugging Face Dataset objects
    features = Features({
        "tokens": Sequence(Value(dtype="string")), # Changed to Sequence(Value(dtype="string")) for lists of tokens
        "ner_tags": Sequence(ClassLabel(names=all_possible_labels)),
    })

    train_dataset = Dataset.from_list(train_data, features=features)
    val_dataset = Dataset.from_list(val_data, features=features)
    test_dataset = Dataset.from_list(test_data, features=features)

    # 4. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    label_to_id = {label: i for i, label in enumerate(all_possible_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(all_possible_labels),
        id2label=id_to_label,
        label2id=label_to_id
    )
    logging.info(f"Model and tokenizer loaded from {MODEL_CHECKPOINT}.")
    logging.info(f"Number of labels: {len(all_possible_labels)}")
    logging.info(f"Label mappings: {label_to_id}")

    # 5. Tokenize and Align Labels for all datasets
    logging.info("Tokenizing and aligning labels for datasets...")
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label_to_id, id_to_label),
        batched=True,
        remove_columns=["tokens", "ner_tags"]
    )
    tokenized_val_dataset = val_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label_to_id, id_to_label),
        batched=True,
        remove_columns=["tokens", "ner_tags"]
    )
    tokenized_test_dataset = test_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label_to_id, id_to_label),
        batched=True,
        remove_columns=["tokens", "ner_tags"]
    )
    logging.info("Tokenization and alignment complete.")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 6. Set up Trainer
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, list(id_to_label.values()))
    )

    # 7. Train the model
    logging.info("Starting model training...")
    trainer.train()
    logging.info("Model training finished.")

    # 8. Evaluate the fine-tuned model on the test set
    logging.info("Evaluating model on the test set...")
    results = trainer.evaluate(tokenized_test_dataset)
    logging.info(f"Test Set Evaluation Results: {results}")

    # 9. Save the model and tokenizer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info(f"✅ Fine-tuned model and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

