# EthioMart/src/data_labeler.py

import pandas as pd
import re
import spacy
from spacy.training import offsets_to_biluo_tags
from pathlib import Path
import logging
from tqdm import tqdm
import os
import random
import sys

# add the project root to sys.path to allow importing from src and config
project_root = Path(__file__).resolve().parent.parent # Points to EthioMart/
sys.path.insert(0, str(project_root)) # Insert at the beginning to prioritize

# Import configurations from config/config.py
from config.config import DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AmharicRuleBasedLabeler:
    """
    A class to perform rule-based entity labeling on Amharic text
    and convert it to CoNLL-like format for NER training.
    
    Entity Types: PRODUCT, PRICE, LOC, CONTACT, DELIVERY
    """
    def __init__(self):
        self.nlp = spacy.blank("xx") # Use a blank model for multilingual (xx) support
        # Refined patterns for better accuracy and to reduce over-tagging.
        # Order of patterns within a list matters, more specific ones first if overlap resolution is simple.
        self.patterns = {
            'PRODUCT': [
                # Specific product names/models (English and Amharic) - Prioritize these exact matches
                r'\b(?:iPhone|Galaxy|Samsung|Huawei|LG|Xiaomi|leather|Tecno|Infinix|Oppo|HP|Dell|Lenovo|Acer|Asus)\s+(?:\d{1,3}[a-zA-Z]?|Pro|Ultra|Plus|Max|Mini|Lite|v\d)\b',
                r'\b(?:Electric Kettle|Automatic Watches|HAND BLENDER|Charcoal Burner|Silicon Lids|Hair Regrowth Treatment|Derma Roller|Bluetooth Speaker|Smartwatch|AirPods|Headphones|Tablet|Camera|Speakers|Power Bank|Smart TV|Blender|Stove|Vibrator|Vacuum Jug|Food mould Tool|Chekich|LITE HANGER|Bathroom soft toilet seat cover|Electronic Kitchen Scale|Red sole Chelsea leather boots|Car Solar Aromatherapy|Multifunctional Drain Rack|Floodlight Head lamp|Bamboo Trey|Portable drying rack Clips cloth hanger|donnut shape cutter|Baby Breast Pads|Threelayer Baby Milk Powder Container|mattress PROTECTOR POLYESTER MICROFIBERBed|Waterproof Running Shoes|Bluetooth Headset|Earphone|Air Cleaner)\b', # Specific English product names observed
                r'\b(?:Converse|Nike|Adidas|Under armour|Skechers|Rolex|MK Watches|CK|EMPORIO ARMANI|VIGUER|NB|Zara|Puma|Jordan)\b(?:\s+\w+){0,3}?\b', # Brand names + optional words for product
                r'\b(?:ስልክ|ሞባይል|ቴሌቪዥን|ላፕቶፕ|ኮምፒውተር|ሰዓት|ጆሮ ማዳመጫ|ሽቶ|ልብስ|ጫማ|ቦርሳ|መኪና|ቤት|መድሃኒት|ምግብ|እቃዎች|ማሽን|ስቶቭ|ብሌንደር|የችብስ መሰንጠቂያ|ጭን ላይ አስቀምጠው መጠቀም|የውሀ ማጣሪያ)\b', # Common Amharic product categories/names
                r'\b(?:3pc|2pc|pc|pcs)\s+\w+(?:\s+\w+){0,2}?\b', # e.g., "3pc silicon brush", "2pc set", "3pc Bottle Stopper"
                r'\b(?:የእግር ትራስ|አንሶላ|ማጣሪያ)\b' # Specific Amharic products from samples
            ],
            'PRICE': [
                # Strict Price patterns with currency unit
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*ETB\b', # e.g., "5500 ETB", "1,500.00 ETB"
                r'\b(?:ዋጋ|ዋጋው|Price|Price is|በ|Discount|ቅናሽ)[:፡]?\s*\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:ብር|ETB|Br|Birr|B\.)?\b', # Prefixed prices with currency
                # Price ranges
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*[-–]\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:ብር|ETB|Br|Birr|B\.)?\b',
                # Numbers followed by "ብር" (birr)
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*ብር\b', 
                # Standalone numbers that are likely prices, contextually (e.g., after "Size" but not part of size range)
                # This needs careful attention to avoid over-tagging sizes as prices.
                # Adding negative lookaheads to prevent matching typical size patterns like "40,41,42"
                r'(?<!Size\s)\b\d{2,}(?:,\d{3})*(?:\.\d+)?\b(?!\s*(?:cm|inch|ml|l|kg|g|\d+))' # Numbers >1 digit, not followed by common units or other numbers
            ],
            'LOC': [
                # Specific well-known locations/areas (English and Amharic)
                r'\b(Addis Ababa|Bole|Mexico|Piyassa|Gerji|Megenagna|Kera|Sarbet|Kazanchis|Piassa|Arada|Lideta|Kolfe|Nifas Silk|Akaki Kaliti|Yeka|Gulele|Jemo|Ayat|22|Mekanisa)\b',
                r'\b(አዲስ አበባ|ቦሌ|መገናኛ|ፒያሳ|ገርጂ|22|ቀበና|ሜክሲኮ|ቂርቆስ|አራዳ|ልደታ|ኮልፌ|ንፋስ ስልክ|አቃቂ ቃሊቲ|የካ|ጉለሌ|ጀሞ|አያት)\b',
                # Address components with prefixes/suffixes
                r'(?:አድራሻ|Address)[:፡]?\s*(?:.+)', # Captures full address phrase
                r'\b(?:ቁ\.?\s*\d+|No\.?\s*\d+)\b', # "ቁ.1", "No. 5" - generic number
                r'\b(?:ፎቅ|floor|ደረጃ)\b', # "ፎቅ", "floor", "ደረጃ"
                r'\b(?:ቢሮ|office)\s*ቁ\.?\s*\S+\b', # "ቢሮ ቁ. S05S06"
                r'\b(?:ቤተ ክርስቲያን|Church|ጀሞ|ጣቢያ)\s*(?:.+)?\b', # Places like churches, stations
                r'\b(?:ሞል|Mall|ሴንተር|Center|plaza|ፕላዛ|ህንፃ|building|ሆቴል)\s*(?:.+)?\b', # Mall/Building names or general terms
                r'\b(?:ፊት ለፊት|Near|አጠገብ|ጀርባ)\b', # "in front of", "near", "behind"
                r'\b(?:ግራውንድ ፍሎር|Ground Floor|1ኛፎቅ|2ተኛ ፎቅ|3 ተኛ ፎቅ|አንደኛ ደረጃ|የመጀመሪያ ደረጃ)\b', # Floor descriptions
                r'\b(?:ጦር ሀይሎች ድሪም ታወር)\b', # Specific building name
                r'\b(?:ራመት ታቦር ኦዳ ህንፃ)\b', # Specific building name
                r'\b(?:አህመድ ህንፃ)\b', # Specific building name
                r'\b(?:መዚድ ፕላዛ)\b', # Specific building name
                r'\b(?:ኮሜርስ)\b', # Specific landmark/area
            ],
            'CONTACT': [
                # Phone numbers (specific formats)
                r'\b(?:\+251|0)?(?:9|7)\d{8}\b', # Covers +2519..., 09..., 9... for 10-digit numbers
                # Telegram usernames
                r'@[\w_]+\b',
                # Email addresses
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                # Contact call-to-actions, followed by actual contact info
                r'(?:ስልክ|Call|ይደውሉ)[:፡]?\s*(?:\+251|0)?(?:9|7)\d{8}\b', # "ስልክ: 09..."
                r'(?:Inbox|መልዕክት|ቴሌግራም ለማዘዝ|ቴሌግራም ገፃችን)[:፡]?\s*@[\w_]+\b', # "Inbox: @username", "ቴሌግራም ገፃችን @username"
            ],
            'DELIVERY': [
                r'\b(?:Free Delivery|ነፃ ዲሊቨሪ|ከነፃ ዲሊቨሪ ጋር|በነፃ ዲሊቨሪ|Delivery Free)\b',
                r'\b(?:ዕቃዉ እጅዎ ሲደርስ|Cash on Delivery|ሲደርስ|when it arrives|በሞባይልባንኪንግ|በካሽ|በካሽ አልያም በሞባይል ባንኪንግ መፈፀም ይችላሉ)\b', # Cash on delivery related phrases
                r'\b(?:ያሉበት ድረስ|doorstep delivery|to your door|እናደርሳለን|እናመጣለን)\b', # To your door phrases/delivery actions
                r'\b(?:ያለተጨማሪ ክፍያ|No additional charge)\b', # For free delivery context
                r'\b(?:የሞተር አናስከፍልም)\b' # "We don't charge for motor delivery" - specific to delivery cost
            ]
        }
    
    def _extract_entities_from_text(self, text):
        """
        Extract entities from the given text using regex patterns.
        Performs basic overlap resolution: prefers earlier, longer matches, then by label priority.
        """ 
        entities = []
        if not isinstance(text, str):
            return entities

        found_matches = []
        for label, patterns in self.patterns.items():
            for pat in patterns:
                for match in re.finditer(pat, text, re.IGNORECASE | re.UNICODE):
                    found_matches.append({
                        'start': match.start(),
                        'end': match.end(),
                        'label': label,
                        'text': match.group(0) # Store matched text for debugging/sorting by length
                    })
        
        # Sort matches for overlap resolution:
        # 1. By start position (earlier matches first)
        # 2. By length (longer matches first - useful for specific vs. general patterns)
        # 3. By a predefined label priority (e.g., PRICE > CONTACT > PRODUCT > LOC > DELIVERY)
        label_priority = {
            'PRICE': 2,
            'CONTACT': 3,
            'DELIVERY': 4,
            'PRODUCT': 5,
            'LOC': 1
        }
        
        found_matches.sort(key=lambda x: (x['start'], -len(x['text']), label_priority.get(x['label'], 0)), reverse=False)

        resolved_entities = []
        
        for match in found_matches:
            start, end, label = match['start'], match['end'], match['label']
            
            is_overlapping = False
            # Fix: Correctly unpack the 3-element tuples from resolved_entities
            for res_start, res_end, _ in resolved_entities: # Unpack 3 elements, ignore the third
                # Check for overlap (start is inclusive, end is exclusive)
                if max(start, res_start) < min(end, res_end):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                resolved_entities.append((start, end, label))
        
        # Ensure the final list is sorted by start offset (important for SpaCy)
        return sorted(resolved_entities, key=lambda x: x[0])


    def process_to_conll(self, input_csv_path, output_conll_path, sample_size=50, random_seed=42):
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
            
            # Filter out empty or NaN preprocessed texts for labeling
            df_filtered = df[df['preprocessed_text'].notna() & (df['preprocessed_text'] != '')]
            logging.info(f"Filtered DataFrame. Rows with non-empty 'preprocessed_text': {len(df_filtered)}")

            if df_filtered.empty:
                logging.warning("No non-empty preprocessed texts found to label. Output CoNLL file will be empty.")
                with open(output_conll_path, 'w', encoding='utf-8') as f:
                    f.write("") # Create an empty file
                return True

            # Sample messages if sample_size is specified
            if sample_size and len(df_filtered) > sample_size:
                texts_to_process_df = df_filtered.sample(n=sample_size, random_state=random_seed).copy()
                logging.info(f"Processing a random sample of {len(texts_to_process_df)} texts.")
            else:
                texts_to_process_df = df_filtered.copy()
                logging.info(f"Processing all {len(texts_to_process_df)} available non-empty texts.")
            
            texts_to_process = texts_to_process_df['preprocessed_text'].tolist()

            num_labeled_messages = 0
            with open(output_conll_path, 'w', encoding='utf-8') as f:
                for text in tqdm(texts_to_process, desc="Labeling texts to CoNLL"):
                    entities = self._extract_entities_from_text(text)
                    doc = self.nlp.make_doc(text) # Tokenize the text using spacy's blank model
                    
                    # spacy.training.offsets_to_biluo_tags converts character-level offsets
                    # to token-level BILUO tags based on the doc's tokenization.
                    tags = offsets_to_biluo_tags(doc, entities)
                    
                    # Ensure tokens and tags match length (should always if offsets are correct)
                    if len(doc) != len(tags):
                        logging.warning(f"Mismatch in token/tag count for text: '{text}'. This message will be skipped for labeling.")
                        f.write("\n") # Add a blank line to signify end of this (skipped) document
                        continue 

                    for token, tag in zip(doc, tags):
                        # Write in CoNLL format: TOKEN \t TAG
                        # If tag is '-', it means no entity found for that token, so 'O' (Outside)
                        f.write(f"{token.text}\t{tag if tag != '-' else 'O'}\n")
                    f.write("\n") # Blank line to separate documents in CoNLL format
                    num_labeled_messages += 1 # Increment only if successfully labeled

            logging.info(f"✅ Successfully labeled and saved {num_labeled_messages} texts to {output_conll_path}")
            return True
            
        except FileNotFoundError:
            logging.error(f"❌ Input file not found: {input_csv_path}. Please ensure preprocessor.py has run and produced the clean_telegram_data.csv.")
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
    output_conll_file = DATA_DIR.parent / "labeled" / "telegram_ner_data_rule_based.conll"

    # You can adjust sample_size here. Keeping at 50 for the initial task.
    # Set to None to process the full dataset.
    labeler.process_to_conll(
        input_csv_path=str(input_csv_file),
        output_conll_path=str(output_conll_file),
        sample_size=50 # Labeling a subset of 50 messages as requested
    )

if __name__ == "__main__":
    main()

