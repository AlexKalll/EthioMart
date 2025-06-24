# EthioMart/src/preprocessor.py

import pandas as pd
import argparse
import re
import logging
import os
import sys
# from emoji import demojize # No longer needed if not converting/demojizing
from pathlib import Path # Ensure Path is imported for path operations

# Ensure the src directory is in the Python path for imports (critical for modules)
project_root = Path(__file__).resolve().parent.parent # Points to EthioMart/
sys.path.insert(0, str(project_root)) # Insert at the beginning to prioritize

# Import paths from your central config.py
from config.config import DATA_DIR 

# Configure logging for the preprocessor script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_amharic(text):
    """
    Applies a series of cleaning and normalization steps to Amharic text from Telegram posts.
    This function aims to make the text clean and consistent for Named Entity Recognition.
    
    This version keeps phone numbers, Telegram usernames, and EMOJIS/ICONS in the text for later labeling.
    """
    if not isinstance(text, str):
        # Ensure input is a string; return empty string for non-string inputs (like NaN)
        return ''

    # Step 1: Normalize common Amharic characters (optional but good practice)
    # This addresses slight Unicode variations or commonly interchanged characters.
    text = text.replace('ሃ', 'ሀ').replace('ሐ', 'ሀ').replace('ሓ', 'ሀ') # Normalizing 'Ha' sounds
    text = text.replace('ጸ', 'ፀ') # Normalizing 'Tse'

    # Step 2: REMOVED ALL EMOJI CLEANING/DEMOJIZING as per user's request
    # Emojis and icons will now remain in the text.
    # The previous code lines that were removed:
    # text = demojize(text, delimiters=("", ""))
    # text = re.sub(r':[a-z_]+:', '', text)
    # emoji_pattern = re.compile([...])
    # text = emoji_pattern.sub(r'', text)

    # Step 3: Clean Telegram-specific decorative patterns and common symbols
    # This list should primarily target non-emoji, non-alphanumeric decorative symbols
    # that are not useful for NER and might clutter the text.
    # Note: Some symbols like pushpin, mobile phone with arrow, etc., might still be Unicode emojis.
    # If any specific Unicode *decorative* symbols (not emojis) are still converted to text or removed
    # that you wish to keep, we can refine this list further.
    telegram_decorative_patterns = [
        # Removed patterns for what are typically emojis/icons if they are to be kept.
        # Keeping patterns for general punctuation/structural elements that are less like emojis.
        r'[\.\s]{3,}', # Ellipsis or multiple dots with spaces (e.g., ... or . . .)
        r'\.{2,}', # Multiple dots (e.g., ..)
        r'\*{2,}',  # Multiple asterisks (e.g., ***)
        r'_{2,}',   # Multiple underscores (e.g., ___)
        r'~{2,}',   # Multiple tildes (e.g., ~~~)
        r'\|{2,}',   # Multiple pipes (e.g., |||)
        r'[\\\/]{2,}', # Multiple backslashes or forward slashes
        r'={2,}', # Multiple equals signs (e.g., ===)
        r'-{2,}', # Multiple hyphens/dashes (e.g., ---)
        r'\+{2,}', # Multiple plus signs (e.g., +++)
        r'#{2,}', # Multiple hash signs (e.g., ###)
        r'%\s*%', # Common "percent" separator
        r'\|', # Single pipe character often used as separator
        r'[\[\]\(\)\{\}]', # Brackets, parentheses, curly braces
        r'\[Image \d+\]', # Remove "[Image X]" if present (from PDF snippets)
        r'\uFFFD' # Unicode replacement character (often appears for unrenderable glyphs)
    ]
    for pattern in telegram_decorative_patterns:
        text = re.sub(pattern, '', text)

    # Step 4: Remove URLs (http/https and t.me links)
    text = re.sub(r'https?://\S+|www\.\S+|t\.me/\S+', '', text)

    # Step 5: Remove Hashtags (Telegram user mentions are kept)
    text = re.sub(r'#\w+', '', text) # Hashtags are still removed

    # Step 6: Standardize currency expressions
    text = re.sub(r'(\d[\d,]*)\s*(ብር|ETB|birr|Br|Birr)', r'\1 ETB', text, flags=re.IGNORECASE)
    text = re.sub(r'[💲🏷💵]', '', text) # Remove these specific currency symbols if not wanted as raw characters
    text = re.sub(r'(\d+),(\d{3})', r'\1\2', text)


    # Step 7: Phone numbers are kept. No specific replacement or removal here.
    # Optional: Remove spaces within phone numbers if you want them contiguous.
    # text = re.sub(r'(\d)\s+(\d)', r'\1\2', text) # Uncomment to remove spaces between digits

    # Step 8: Replace multiple spaces with a single space and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 9: Remove any remaining non-Amharic/non-English letters/digits characters
    # This regex now permits:
    # Amharic Unicode range (\u1200-\u137F)
    # Digits (0-9)
    # English letters (a-zA-Z)
    # Common punctuation (.,!?;:)
    # Whitespace (\s)
    # The '@' symbol (for Telegram usernames)
    # The '+' symbol (for phone numbers like +251)
    text = re.sub(r'[^\u1200-\u137F0-9a-zA-Z.፣፤!:\s@+።]', '', text)
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
        'channel_title', 'message_id', 'date', 'preprocessed_text',
        'views', 'reactions_count'
    ]
    
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        logging.error(f"❌ Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    if df['message_id'].isnull().any():
        logging.error("❌ NULL values found in 'message_id' column.")
        raise ValueError("NULL values in 'message_id' column.")
    
    if (df['preprocessed_text'].isnull()).any() or (df['preprocessed_text'] == '').any():
        logging.warning("⚠️ Some 'preprocessed_text' entries are empty or null.")

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
            logging.error(f"❌ Error loading input CSV {args.input}: {e}", exc_info=True) # Added exc_info for full traceback
            exit()

        # Apply preprocessing
        df['preprocessed_text'] = df['text'].fillna('').apply(preprocess_amharic)
        logging.info(f"Preprocessing applied. Sample preprocessed text:\n{df['preprocessed_text'].head()}")
        
        # Select only the desired columns for the output CSV
        output_columns = [
            'channel_title', 'message_id', 'date', 'preprocessed_text',
            'views', 'reactions_count'
        ]
        # Ensure all output_columns exist in df before selection
        if not all(col in df.columns for col in output_columns):
            missing = [col for col in output_columns if col not in df.columns]
            logging.error(f"❌ Cannot save: Missing required columns for output: {missing}")
            exit()

        df_output = df[output_columns].copy() # Create a copy to avoid SettingWithCopyWarning

        # Define the output directory
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

        # Save the processed DataFrame
        df_output.to_csv(args.output, index=False, encoding='utf-8')
        
        logging.info(f"✅ Saved {len(df_output)} preprocessed messages to {args.output}")

