# EthioMart/src/preprocessor.py

import pandas as pd
import argparse
import re
import logging
import os
import sys
from pathlib import Path

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
    
    This version keeps phone numbers, Telegram usernames, and emojis/icons in the text for later labeling.
    It includes significantly improved tokenization by adding spaces around key punctuation and mixed scripts.
    """
    if not isinstance(text, str):
        return ''

    # Step 1: Normalize common Amharic characters
    text = text.replace('ሃ', 'ሀ').replace('ሐ', 'ሀ').replace('ሓ', 'ሀ')
    text = text.replace('ጸ', 'ፀ')

    # Step 2: Remove all emojis and a broad range of pictorial/decorative symbols
    # This uses a comprehensive regex pattern to target various Unicode blocks.
    emoji_and_symbol_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
        "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed Characters, Alphanumeric, etc.
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\u200d"                 # Zero-width joiner
        "\uFE0F"                 # Variation selector
        # Comprehensive range of common symbols and dingbats that are not text
        "\U00002000-\U0000206F"  # General Punctuation (e.g., thin space, hair space) - cautious with this range
        "\U00002100-\U0000214F"  # Letterlike Symbols
        "\U00002190-\U000021FF"  # Arrows
        "\U00002300-\U000023FF"  # Miscellaneous Technical
        "\U000025A0-\U000025FF"  # Geometric Shapes
        "\U000027F0-\U000027FF"  # Supplemental Arrows-A
        "\U00002900-\U0000297F"  # Supplemental Arrows-B
        "\U00002B00-\U00002BFF"  # Miscellaneous Symbols and Arrows
        "\U00003000-\U0000303F"  # CJK Symbols and Punctuation (might have some decorative)
        "\U0001F000-\U0001F02F"  # Mahjong Tiles / Domino Tiles
        "\U0001F0A0-\U0001F0FF"  # Playing Cards
        "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs (already included some)
        "\U0001F600-\U0001F64F"  # Emoticons (already included)
        "\U0001F680-\U0001F6FF"  # Transport and Map Symbols (already included)
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs (already included)
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FAB0-\U0001FABF"  # AVD Symbols
        "\U0001FAC0-\U0001FACF"  # Face Symbols
        "\U0001FAD0-\U0001FADF"  # Plant Symbols
        "\U00002B50"             # Black star
        "]+", flags=re.UNICODE
    )
    text = emoji_and_symbol_pattern.sub(r'', text)
    
    # Step 3: Clean Telegram-specific general decorative patterns (non-emoji/pictorial)
    telegram_decorative_patterns = [
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
    text = re.sub(r'[💲🏷💵]', '', text) 
    text = re.sub(r'(\d+),(\d{3})', r'\1\2', text) # Remove commas within numbers


    # Step 7: Phone numbers are kept.
    # Optional: Remove spaces within phone numbers if you want them contiguous.
    # text = re.sub(r'(\d)\s+(\d)', r'\1\2', text) # Uncomment to remove spaces between digits

    # Step 7.5: Insert spaces around common punctuation marks and delimiters
    # This is CRITICAL for better SpaCy tokenization and separating concatenated words.
    # Amharic punctuation: (።) full stop, (፣) comma, (፤) semicolon, (፡) colon
    # English punctuation: (.), (,), (!), (?), (:), (;)
    # Add spaces around these if they are stuck to words, but not if already spaced.
    text = re.sub(r'(?<=\S)([.,!?;:፡፣፤።])(?=\S)', r' \1 ', text)
    
    # Add spaces between Amharic and English words/digits if concatenated
    text = re.sub(r'([\u1200-\u137F])([a-zA-Z0-9@+])', r'\1 \2', text) # Amharic followed by English/Digit/@/+
    text = re.sub(r'([a-zA-Z0-9@+])([\u1200-\u137F])', r'\1 \2', text) # English/Digit/@/+ followed by Amharic

    # Add spaces between digits and letters if concatenated (e.g., '101የቢሮ')
    text = re.sub(r'(\d)([a-zA-Z\u1200-\u137F])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z\u1200-\u137F])(\d)', r'\1 \2', text)
    
    # Add spaces around '@' and '+' if they are not already separated by space
    text = re.sub(r'(?<=\S)(@)(?=\S)', r' \1 ', text)
    text = re.sub(r'(?<=\S)(\+)(?=\S)', r' \1 ', text)


    # Step 8: Replace multiple spaces with a single space and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 9: Remove any remaining non-Amharic/non-English letters/digits characters
    # This regex now permits:
    # Amharic Unicode range (\u1200-\u137F)
    # Digits (0-9)
    # English letters (a-zA-Z)
    # Common punctuation (.,!?;:), also Amharic specific (።,፣,፤,፡)
    # Whitespace (\s)
    # The '@' symbol (for Telegram usernames)
    # The '+' symbol (for phone numbers like +251)
    # This ensures anything not explicitly allowed is removed.
    text = re.sub(r'[^\u1200-\u137F0-9a-zA-Z.,!?;:፡፣፤።\s@+]', '', text)
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
            logging.error(f"❌ Error loading input CSV {args.input}: {e}", exc_info=True)
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

        df_output = df[output_columns].copy()

        # Define the output directory
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the processed DataFrame
        df_output.to_csv(args.output, index=False, encoding='utf-8')
        
        logging.info(f"✅ Saved {len(df_output)} preprocessed messages to {args.output}")

