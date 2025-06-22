# EthioMart/src/data_labeler.py

import pandas as pd
import re
import spacy
from spacy.training import offsets_to_biluo_tags
from pathlib import Path
import logging
from tqdm import tqdm
import os

# Import paths from your central config.py
try:
    from config.config import DATA_DIR
except ImportError:
    logging.error("Could not import DATA_DIR from config.config. Please ensure the config file is correct.")
    # Fallback for testing/debugging outside the main project structure if config isn't set up
    DATA_DIR = Path(__file__).parent.parent / "data"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AmharicRuleBasedLabeler:
    """
    A class to perform rule-based entity labeling on Amharic text
    and convert it to CoNLL-like format for NER training.
    """
    def __init__(self):
        # Using a blank 'xx' (multilingual) model for tokenization
        # We are not using a pre-trained model for entity recognition at this stage,
        # but merely for tokenization to align with CoNLL format.
        self.nlp = spacy.blank("xx")
        
        # Comprehensive Regex Patterns for Amharic E-commerce Entities
        self.patterns = {
            'PRICE': [
                # Matches numbers with optional commas, followed by common currency indicators
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:ብር|ETB|birr|Br|Birr|B\.)\b',
                # Matches prices starting with "Price:", "ዋጋ", "ዋጋው", etc.
                r'(?:Price|ዋጋ|ዋጋው)[:፡]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:ብር|ETB|birr|Br|Birr|B\.)?\b',
                # Matches price ranges like "500-1000 ብር"
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*[-–]\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:ብር|ETB|birr|Br|Birr|B\.)?\b',
            ],
            'LOC': [
                # Specific common locations in Amharic/English (expand as needed)
                r'\b(Addis Ababa|Bole|Mexico|Piyassa|Gerji|Megenagna|Kera|Sarbet|Kazanchis|Piassa|Arada|Lideta|Kolfe|Nifas Silk|Akaki Kaliti|Yeka|Gulele)\b',
                r'\b(አዲስ አበባ|ቦሌ|መገናኛ|ፒያሳ|ገርጂ|22|ቀበና|ሜክሲኮ|ቂርቆስ|አራዳ|ልደታ|ኮልፌ|ንፋስ ስልክ|አቃቂ ቃሊቲ|የካ|ጉለሌ)\b',
                # Addresses with "አድራሻ", "ቁጥር", "ፎቅ", "floor"
                r'(?:አድራሻ|Address)[:፡]?\s*(.+)', # Captures everything after "አድራሻ"
                r'\b(?:ቁጥር|No|floor|ፎቅ)\s*\d+\b', # Matches "ቁጥር 1", "No. 5", "3rd floor"
                r'\b(?:ቤተ ክርስቲያን|Church)\s*(.+)\b', # For locations near churches
                r'\b(?:ሞል|Mall)\s*(.+)?\b', # For locations inside malls
                r'\b(?:ፊት ለፊት)\b' # "in front of"
            ],
            'PRODUCT': [
                # Common product keywords and patterns, potentially with emojis or descriptions
                r'(?:iPhone|Galaxy|Samsung|Huawei|LG|Xiaomi|Tecno|Infinix|Oppo)\s+(?:\d{1,2}|Pro|Ultra|Plus|Max)\b', # Specific phone models
                r'\b(ስልክ|ሞባይል|ቴሌቪዥን|ላፕቶፕ|ኮምፒውተር|ሰዓት|ጆሮ ማዳመጫ|ሽቶ|ልብስ|ጫማ|ቦርሳ|መኪና|ቤት|መድሃኒት|ምግብ|እቃዎች|ማሽን|Humidifier|Smartwatch|AirPods|Headphones|Tablet|Camera|Speakers|Power Bank|Smart TV)\b',
                r'[\U0001F4CD\U0001F4C6]\s*(.+)', # 📌📍 emoji followed by product (or anything, needs context)
                r'\b\w+\s*Vibrator\b', # Example from labeled data
                r'\b(?:3pc|2pc|pc)\s+\w+\s+\w+\b', # e.g., "3pc silicon brush"
                r'\b(?:Slimming Belt|Mandoline Slicer)\b' # Examples from labeled data
            ],
            'CONTACT': [
                # Phone numbers: various formats, including local and international prefixes
                r'\b(?:\+251|0)?(?:9|7)\d{8}\b', # Covers +2519..., 09..., 9...
                r'\b\d{2}[\s-]?\d{3}[\s-]?\d{4}\b', # For XX XXX XXXX format (e.g., 09 111 2233)
                # Telegram usernames
                r'@[\w_]+\b',
                # Email addresses
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                # Contact phrases
                r'(?:ስልክ|Contact|ይደውሉ|Call)[:፡]?\s*(.+)' # Captures text after contact phrase
            ]
        }

    def _extract_entities_from_text(self, text):
        """
        Extracts entities from a given text using defined regex patterns.
        Returns a list of (start_offset, end_offset, label) tuples.
        """
        entities = []
        # Ensure text is a string to prevent regex errors
        if not isinstance(text, str):
            return entities

        for label, patterns in self.patterns.items():
            for pat in patterns:
                for match in re.finditer(pat, text, re.IGNORECASE):
                    # For patterns that capture a group (e.g., r'📍\s*(.+)')
                    # we want the entity to be the captured group, not the full match.
                    # This requires careful handling for specific patterns.
                    # For simplicity and to avoid over-tagging non-entity parts of a match,
                    # we will mostly rely on the full match span for now.
                    # For patterns like r'(?:አድራሻ|Address)[:፡]?\s*(.+)', the entity is the captured group.
                    # However, spacy's offsets_to_biluo_tags works on the original string's offsets.
                    # So, `match.start()` and `match.end()` are correct for the full matched span.
                    
                    # If a pattern uses a capturing group for the *actual entity*,
                    # e.g., r'ዋጋ\s*[:፡]\s*(\d+)', then match.start(1) and match.end(1)
                    # would be used. But for general patterns, full span is safer.
                    # The current patterns mostly match the full entity already.

                    entities.append((match.start(), match.end(), label))
        
        # Sort entities by their start offset to handle overlaps and ensure correct BILUO tagging
        # In case of overlaps, the longest match or the first one found could be prioritized.
        # Spacy's `offsets_to_biluo_tags` handles overlaps by prioritizing earlier/longer.
        return sorted(entities, key=lambda x: x[0])

    def process_to_conll(self, input_csv_path, output_conll_path, sample_size=None):
        """
        Loads preprocessed data, extracts entities, and writes them to a CoNLL-like format.
        """
        logging.info(f"Starting rule-based labeling from {input_csv_path}...")
        
        # Ensure output directory exists
        output_dir = Path(output_conll_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_csv(input_csv_path, encoding='utf-8')
            logging.info(f"Loaded DataFrame from {input_csv_path}. Total rows: {len(df)}")
            
            # Filter out empty or NaN preprocessed texts
            df_filtered = df[df['preprocessed_text'].notna() & (df['preprocessed_text'] != '')]
            logging.info(f"Filtered DataFrame. Rows with non-empty 'preprocessed_text': {len(df_filtered)}")

            texts = df_filtered['preprocessed_text'].tolist()
            
            if sample_size:
                texts = texts[:sample_size]
                logging.info(f"Processing a sample of {len(texts)} texts.")
            
            with open(output_conll_path, 'w', encoding='utf-8') as f:
                for text in tqdm(texts, desc="Labeling texts to CoNLL"):
                    entities = self._extract_entities_from_text(text)
                    doc = self.nlp.make_doc(text)
                    
                    # spacy.training.offsets_to_biluo_tags expects (start, end, label)
                    # It will convert these character offsets into token-level BILUO tags.
                    tags = offsets_to_biluo_tags(doc, entities)
                    
                    for token, tag in zip(doc, tags):
                        # Write in CoNLL format: TOKEN \t TAG
                        # If tag is '-', it means no entity found for that token, so 'O' (Outside)
                        f.write(f"{token.text}\t{tag if tag != '-' else 'O'}\n")
                    f.write("\n") # Blank line to separate documents in CoNLL format
            
            logging.info(f"✅ Successfully labeled and saved {len(texts)} texts to {output_conll_path}")
            return True
            
        except FileNotFoundError:
            logging.error(f"❌ Input file not found: {input_csv_path}. Please ensure preprocessor.py has run.")
            return False
        except pd.errors.EmptyDataError:
            logging.error(f"❌ Input file {input_csv_path} is empty. No data to label.")
            return False
        except Exception as e:
            logging.error(f"❌ Error during labeling process: {e}", exc_info=True)
            return False

def main():
    labeler = AmharicRuleBasedLabeler()
    
    # Define input and output paths using DATA_DIR from config
    input_csv_file = DATA_DIR.parent / "processed" / "clean_telegram_data.csv"
    output_conll_file = DATA_DIR.parent / "annotated" / "telegram_ner_data_rule_based.conll"

    # You can adjust sample_size or remove it to process the full dataset
    labeler.process_to_conll(
        input_csv_path=str(input_csv_file),
        output_conll_path=str(output_conll_file),
        sample_size=None # Set to an integer like 1000 for a sample, or None for full dataset
    )

if __name__ == "__main__":
    main()
