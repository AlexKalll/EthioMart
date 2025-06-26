# EthioMart/src/vendor_scorecard_engine.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import re
import torch
from datetime import datetime, timedelta
from tqdm import tqdm

# Hugging Face imports for NER inference
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Add the project root to sys.path to allow importing from src and config
project_root = Path(__file__).resolve().parent.parent # Points to EthioMart/
sys.path.insert(0, str(project_root)) # Insert at the beginning to prioritize

# Import configurations and preprocessor
from config.config import DATA_DIR
from src.preprocessor import preprocess_amharic

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
CLEAN_DATA_PATH = Path(DATA_DIR.parent / "processed" / "clean_telegram_data.csv")
BEST_MODEL_PATH = Path(project_root / "models" / "distilbert_ner_fine_tuned") # Use the best model (DistilBERT)
OUTPUT_PATH = Path(project_root / "outputs" / "vendor_scorecard.csv")

class VendorScorecardEngine:
    def __init__(self):
        self.data = self._load_data()
        self.nlp_pipeline = self._load_ner_model()
        
        if self.data is None:
            logging.error("Failed to load clean data. Exiting VendorScorecardEngine initialization.")
            sys.exit(1)
        if self.nlp_pipeline is None:
            logging.error("Failed to load NER model. Exiting VendorScorecardEngine initialization.")
            sys.exit(1)

        # Ensure 'date' column is datetime type for frequency calculation
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
        else:
            logging.error("Date column not found in data. Cannot calculate posting frequency.")
            sys.exit(1)

        # Apply NER to all preprocessed texts
        logging.info("Running NER inference on all preprocessed texts (this may take a while)...")
        # Initialize an empty list to store all predictions
        all_predictions = []
        for text in tqdm(self.data['preprocessed_text'].fillna(' ').tolist(), desc="NER Inference"):
            # The pipeline expects a single string for prediction, not a list of words.
            # Make sure preprocess_amharic returns a string.
            predictions = self.nlp_pipeline(text)
            all_predictions.append(predictions)
        self.data['ner_predictions'] = all_predictions
        logging.info("NER inference complete for all messages.")

    def _load_data(self):
        """Loads the clean telegram data."""
        if not CLEAN_DATA_PATH.exists():
            logging.error(f"Clean data file not found at {CLEAN_DATA_PATH}. Please run preprocessor.py.")
            return None
        try:
            df = pd.read_csv(CLEAN_DATA_PATH)
            logging.info(f"Loaded clean data from {CLEAN_DATA_PATH}. Total rows: {len(df)}")
            return df
        except Exception as e:
            logging.error(f"Error loading clean data: {e}")
            return None

    def _load_ner_model(self):
        """Loads the fine-tuned NER model and tokenizer."""
        if not BEST_MODEL_PATH.exists():
            logging.error(f"Best model not found at {BEST_MODEL_PATH}. Please run model_finetuner.py or distilbert_finetuner.py.")
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_PATH)
            model = AutoModelForTokenClassification.from_pretrained(BEST_MODEL_PATH)
            device = 0 if torch.cuda.is_available() else -1
            nlp_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device
            )
            logging.info(f"Loaded NER model from {BEST_MODEL_PATH} for inference.")
            return nlp_pipeline
        except Exception as e:
            logging.error(f"Error loading NER model: {e}")
            return None

    def _extract_entity(self, predictions, entity_type):
        """Extracts the first occurrence of a specific entity type from predictions."""
        for pred in predictions:
            if pred['entity_group'] == entity_type:
                return pred['word']
        return None

    def _calculate_metrics(self, vendor_df):
        """Calculates key metrics for a single vendor."""
        metrics = {}

        # Activity & Consistency: Posting Frequency
        if not vendor_df.empty:
            min_date = vendor_df['date'].min()
            max_date = vendor_df['date'].max()
            duration_days = (max_date - min_date).days
            num_posts = len(vendor_df)

            if duration_days > 0:
                metrics['Posting_Frequency_per_Week'] = (num_posts / duration_days) * 7
            else: # If all posts are on the same day, consider it 1 post per week for simplicity
                metrics['Posting_Frequency_per_Week'] = num_posts * 7 # e.g., 1 post on one day = 7 posts/week if only 1 day data
        else:
            metrics['Posting_Frequency_per_Week'] = 0

        # Market Reach & Engagement: Average Views per Post
        metrics['Average_Views_per_Post'] = vendor_df['views'].mean() if not vendor_df.empty else 0

        # Top Performing Post
        top_post = vendor_df.loc[vendor_df['views'].idxmax()] if not vendor_df.empty else None
        if top_post is not None:
            top_post_predictions = top_post['ner_predictions']
            metrics['Top_Product'] = self._extract_entity(top_post_predictions, 'PRODUCT')
            metrics['Top_Price'] = self._extract_entity(top_post_predictions, 'PRICE')
        else:
            metrics['Top_Product'] = None
            metrics['Top_Price'] = None

        # Business Profile: Average Price Point
        all_prices = []
        for predictions in vendor_df['ner_predictions']:
            for pred in predictions:
                if pred['entity_group'] == 'PRICE':
                    # Extract numeric part, assume ETB and convert
                    price_str = re.sub(r'[^\d.]', '', pred['word']) # Keep only digits and decimal
                    try:
                        all_prices.append(float(price_str))
                    except ValueError:
                        continue
        metrics['Average_Price_Point_ETB'] = np.mean(all_prices) if all_prices else 0

        return metrics

    def generate_scorecard(self):
        """Generates a scorecard for all unique vendors."""
        vendor_channels = self.data['channel_title'].unique()
        scorecard_data = []

        for channel_title in tqdm(vendor_channels, desc="Generating Vendor Scorecards"):
            vendor_df = self.data[self.data['channel_title'] == channel_title].copy()
            
            # Recalculate 'Posting_Frequency_per_Week' for vendors with very few posts
            # If a vendor has only 1 post, duration_days would be 0, leading to issues.
            # Let's adjust this to avoid division by zero or overly high frequency.
            if len(vendor_df) == 1:
                logging.info(f"Vendor '{channel_title}' has only one post. Setting frequency to 1 post/week.")
                posting_frequency = 1.0 # Or some other reasonable default for a single post
            elif len(vendor_df) > 1:
                min_date = vendor_df['date'].min()
                max_date = vendor_df['date'].max()
                duration_days = (max_date - min_date).days
                if duration_days > 0:
                    posting_frequency = (len(vendor_df) / duration_days) * 7
                else: # All posts on the same day, but more than one post
                    posting_frequency = len(vendor_df) * 7 # Treat as if all happened in one week
            else: # No posts for this vendor (shouldn't happen if `vendor_channels` is derived from `self.data`)
                posting_frequency = 0.0

            metrics = self._calculate_metrics(vendor_df)
            metrics['Posting_Frequency_per_Week'] = posting_frequency # Override with robust freq

            # Create a simple "Lending Score"
            # Define weights for each metric. These are design choices.
            # You can adjust these weights based on business priorities.
            avg_views_weight = 0.5
            posting_freq_weight = 0.5
            
            # Normalize metrics if necessary, especially if values vary widely.
            # For simplicity in this initial version, we'll use raw values.
            
            # Handle potential None values for product/price from NER
            lending_score = (metrics['Average_Views_per_Post'] * avg_views_weight) + \
                            (metrics['Posting_Frequency_per_Week'] * posting_freq_weight)
            
            # Add vendor channel name
            metrics['Vendor_Channel'] = channel_title
            metrics['Lending_Score'] = lending_score
            
            scorecard_data.append(metrics)

        scorecard_df = pd.DataFrame(scorecard_data)
        
        # Reorder columns for clarity
        cols = ['Vendor_Channel', 'Posting_Frequency_per_Week', 'Average_Views_per_Post',
                'Top_Product', 'Top_Price', 'Average_Price_Point_ETB', 'Lending_Score']
        scorecard_df = scorecard_df[cols]
        
        return scorecard_df

    def display_scorecard(self, scorecard_df):
        """Prints the scorecard in a readable format."""
        logging.info("\n--- FinTech Vendor Scorecard ---")
        print(scorecard_df.to_string(index=False))
        # and save it to a CSV file
        scorecard_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
        logging.info("\nScorecard generation complete.")
        logging.info(f"VendorScorecard saved to {OUTPUT_PATH}")

def main():
    engine = VendorScorecardEngine()
    scorecard = engine.generate_scorecard()
    engine.display_scorecard(scorecard)

if __name__ == "__main__":
    main()

