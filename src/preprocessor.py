# EthioMart/src/preprocessor.py

import pandas as pd
import argparse
import re
import logging
import os
import sys
from emoji import demojize # For handling emojis
from pathlib import Path # Ensure Path is imported for path operations

# Ensure the src directory is in the Python path for imports
project_root = Path(__file__).resolve().parent.parent # Points to EthioMart/
sys.path.insert(0, str(project_root)) # Insert at the beginning to prioritize

# Import paths from your central config.py
from config.config import DATA_DIR # Assuming DATA_DIR points to EthioMart/data/raw

# Configure logging for the preprocessor script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_amharic(text):
    """
    Applies a series of cleaning and normalization steps to Amharic text from Telegram posts.
    This function aims to make the text clean and consistent for Named Entity Recognition.
    """
    if not isinstance(text, str):
        # Ensure input is a string; return empty string for non-string inputs (like NaN)
        return ''

    # Step 1: Normalize common Amharic characters (optional but good practice)
    # This addresses slight Unicode variations or commonly interchanged characters.
    text = text.replace('ሃ', 'ሀ').replace('ሐ', 'ሀ').replace('ሓ', 'ሀ') # Normalizing 'Ha' sounds
    text = text.replace('ሰ', 'ሠ') # Normalizing 'Se' and 'She'
    text = text.replace('ጸ', 'ፀ') # Normalizing 'Tse'

    # Step 2: Remove all emojis and their text representations (e.g., :smile:)
    text = demojize(text, delimiters=("", "")) # Convert to shortcodes, then remove
    text = re.sub(r':[a-z_]+:', '', text) # Remove any remaining shortcodes
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)

    # Step 3: Clean Telegram-specific decorative patterns and common symbols
    telegram_decorative_patterns = [
        r'[\U00002600-\U000026FF]', # Miscellaneous Symbols (e.g., 📌📍)
        r'[\U00002700-\U000027BF]', # Dingbats (e.g., ✅❌)
        r'[\U00002B50]', # Star symbol
        r'[\U00002B00-\U00002BFF]', # Arrow symbols (e.g., ⤵️🔽)
        r'[👍⚡️⚠️🏢🔖💬]', # Common Telegram icons and hand gestures
        r'[\u200b\u200c\u200d\u200e\u200f]', # Zero-width joiner, non-joiner, etc. (hidden characters)
        r'[🔸♦️✨✔️🤍🔶⭐️🌟🔥💧]', # Other decorative symbols seen in samples
        r'\[Image \d+\]', # Remove "[Image X]" if present (from PDF snippets)
        r'[\.\s]{3,}', # Ellipsis or multiple dots with spaces (e.g., ... or . . .)
        r'\.{2,}', # Multiple dots (e.g., ..)
        r'+',      # Replacement characters for unrenderable glyphs
        r'\*{2,}',  # Multiple asterisks (e.g., ***)
        r'_{2,}',   # Multiple underscores (e.g., ___)
        r'~{2,}',   # Multiple tildes (e.g., ~~~)
        r'\|{2,}',   # Multiple pipes (e.g., |||)
        r'[\\\/]{2,}', # Multiple backslashes or forward slashes
        r'={2,}', # Multiple equals signs (e.g., ===)
        r'-{2,}', # Multiple hyphens/dashes (e.g., ---)
        r'\+{2,}', # Multiple plus signs
        r'#{2,}', # Multiple hash signs
        r'%\s*%', # Common "percent" separator
        r'\|', # Single pipe character often used as separator
        r'[\[\]\(\)\{\}]', # Brackets, parentheses, curly braces
    ]
    for pattern in telegram_decorative_patterns:
        text = re.sub(pattern, '', text)

    # Step 4: Remove URLs (http/https and t.me links)
    text = re.sub(r'https?://\S+|www\.\S+|t\.me/\S+', '', text)

    # Step 5: Remove Telegram user mentions (@username) and hashtags (#hashtag)
    text = re.sub(r'@\w+', '', text) # Telegram mentions
    text = re.sub(r'#\w+', '', text) # Hashtags

    # Step 6: Standardize currency expressions
    text = re.sub(r'(\d[\d,]*)\s*(ብር|ETB|birr|Br|Birr)', r'\1 ETB', text, flags=re.IGNORECASE)
    text = re.sub(r'[💲🏷💵]', '', text)
    text = re.sub(r'(\d+),(\d{3})', r'\1\2', text)


    # Step 7: Clean and standardize phone numbers
    text = re.sub(r'(\+251\s?\d{9}|\d{2}\s?\d{3}\s?\d{4}|\d{9})', r' <PHONE_NUMBER> ', text)


    # Step 8: Replace multiple spaces with a single space and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 9: Remove any remaining non-Amharic/non-English letters/digits characters
    text = re.sub(r'[^\u1200-\u137F0-9a-zA-Z.,!?;:\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip() # Re-strip after potential new spaces from regex

    return text

def validate_csv(input_path):
    """
    Validates the structure and content of the processed CSV file.
    """
    logging.info(f"Validating CSV file: {input_path}")
    if not os.path.exists(input_path):
        logging.error(f"❌ Error: CSV file not found at {input_path}")
        raise FileNotFoundError(f"Processed CSV file not found: {input_path}")

    df = pd.read_csv(input_path, encoding='utf-8')
    
    required_columns = [
        'channel_title', 'message_id', 'date', 'text',
        'views', 'reactions_count', 'image_path', 'preprocessed_text'
    ]
    
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        logging.error(f"❌ Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    if df['message_id'].isnull().any():
        logging.error("❌ NULL values found in 'message_id' column.")
        raise ValueError("NULL values in 'message_id' column.")
    
    messages_with_original_text = df[df['text'].notnull() & (df['text'] != '')]
    if (messages_with_original_text['preprocessed_text'].isnull()).any() or \
       (messages_with_original_text['preprocessed_text'] == '').any():
        logging.warning("⚠️ Some messages that originally had text resulted in empty or null preprocessed_text.")

    logging.info("✅ CSV validation passed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Telegram text data.")
    parser.add_argument('--validate', action='store_true',
                        help='Run validation on the output CSV instead of processing.')
    parser.add_argument('--input', type=str,
                        default=str(DATA_DIR / "telegram_data.csv"), # Use Path objects for default
                        help='Path to the input raw CSV file from scraper.')
    parser.add_argument('--output', type=str,
                        default=str(DATA_DIR.parent / "processed" / "clean_telegram_data.csv"),
                        help='Path for the output preprocessed CSV file.')
    
    args = parser.parse_args()

    if args.validate:
        validate_csv(args.output)
    else:
        logging.info(f"Starting preprocessing from {args.input}...")
        
        # Load the raw data
        try:
            df = pd.read_csv(args.input, encoding='utf-8')
            logging.info(f"Loaded DataFrame from {args.input}. Initial rows: {len(df)}")
            if df.empty:
                logging.warning("The loaded DataFrame is empty. This might be why no data is being processed.")
            else:
                logging.info(f"First 5 rows of 'text' column:\n{df['text'].head()}")
        except FileNotFoundError:
            logging.error(f"❌ Input file not found: {args.input}. Please ensure the scraper has run successfully and populated data/raw/telegram_data.csv.")
            exit()
        except pd.errors.EmptyDataError:
            logging.error(f"❌ Input file {args.input} is empty. No data to preprocess. Please check scraper output.")
            exit()
        except Exception as e:
            logging.error(f"❌ Error loading input CSV {args.input}: {e}")
            exit()

        # Apply preprocessing
        df['preprocessed_text'] = df['text'].fillna('').apply(preprocess_amharic)
        logging.info(f"Preprocessing applied. Sample preprocessed text:\n{df['preprocessed_text'].head()}")
        
        # Define the output directory
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

        # Save the processed DataFrame
        df.to_csv(args.output, index=False, encoding='utf-8')
        
        logging.info(f"✅ Saved {len(df)} preprocessed messages to {args.output}")

