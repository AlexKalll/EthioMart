# EthioMart | Amharic E-commerce Data Extractor

## 📘 Project Overview

EthioMart aims to become the primary hub for Telegram-based e-commerce in Ethiopia by centralizing real-time data from various channels. This project focuses on building an Amharic Named Entity Recognition (NER) system to extract key business entities (product names, prices, locations, contacts) from Telegram messages and images. The extracted data will populate EthioMart's centralized database, enabling a seamless experience for customers and informing FinTech initiatives like vendor loan assessments.

This project leverages Python, Telegram API, and data science tools to build a robust data pipeline, from scraping and preprocessing to model fine-tuning and interpretability.

## 🏗️ 1. Project Setup and Data Collection (Task 1)

**Deliverables:**

* GitHub code for Task 1 (data ingestion and preprocessing).
* Data summary (1-2 pages) covering data preparation and labeling steps.

### Repo/Project Structure

```bash
EthioMart/
├── src/
│   ├── telegram_scraper.py # Collects raw data from Telegram channels
│   ├── preprocessor.py     # Cleans and preprocesses raw text data
│   ├── data_labeler.py     # Rule-based labeling for NER (Task 2)
│   └── model_finetuner.py  # Fine-tunes NER models (Task 3 & 4)
├── config/
│   └── config.py           # Stores configuration variables (e.g., API credentials, channel list)
├── data/
│   ├── raw/                # Stores raw scraped data (e.g., telegram_data.csv)
│   │   └── telegram_data.csv
│   ├── processed/          # Stores cleaned and preprocessed data
│   │   └── clean_telegram_data.csv
│   └── labeled/            # Stores manually and semi-automatically labeled data
│       └── telegram_ner_data_rule_based.conll
├── models/                 # Stores fine-tuned NER models (Task 3 & 4)
│   └── afro_xlmr_ner_fine_tuned/
├── photos/                 # Stores downloaded images from Telegram messages
├── notebooks/              # Jupyter notebooks for EDA, experimentation, and documentation
│   ├── data_ingestion_eda.ipynb
│   └── data_preprocessing_eda.ipynb
├── outputs/                # Stores generated plots and visualizations
│   └── plots/
├── reports/                # For interim and final project reports
├── tests/                  # Unit tests for various modules (e.g., preprocessor)
│   └── test_preprocessor.py
│   └── test_telegram_scraper.py
├── .github/workflows/      # CI/CD pipelines (e.g., for DVC and code quality)
├── .env                    # Environment variables (e.g., Telegram API keys)
├── requirements.txt        # Python package dependencies
├── .gitignore              # Files/directories to ignore in Git
├── README.md               # Project overview and setup instructions
```

### Tech Stack - Tools Used

* **Python 3.11+**
* **Telethon:** For interacting with the Telegram API to scrape messages and metadata.
* **Pandas, NumPy:** For efficient data manipulation and analysis.
* **Matplotlib, Seaborn:** For data visualization and exploratory data analysis.
* **Jupyter Notebook:** For interactive data exploration and reproducible analysis.
* **re (Regex):** For advanced text cleaning and pattern matching.
* **pathlib:** For robust path management.
* **pytest:** For unit testing the project's functions.
* **Hugging Face transformers:** For loading, fine-tuning, and evaluating transformer models.
* **Hugging Face datasets:** For efficient data loading and preprocessing for LLMs.
* **seqeval:** For evaluating NER model performance.
* **torch (PyTorch):** The deep learning framework used by the models.
* **scikit-learn:** For data splitting utilities.
* **tensorboard:** For visualizing training progress.

## 🚀 2. Usage and Data Pipeline Steps

This section guides you through the process of setting up the project, collecting data, and performing initial analysis.

### Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/AlexKalll/EthioMart.git
    cd EthioMart
    ```

2.  **Set Up Environment Variables:**
    Create a `.env` file in the project root:

    ```bash
    TELEGRAM_API_ID=your_api_id
    TELEGRAM_API_HASH=your_api_hash
    TELEGRAM_PHONE_NUMBER=your_phone_number
    ```

    Obtain `API_ID` and `API_HASH` from [my.telegram.org](https://my.telegram.org).

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Data Pipeline Steps

1.  **Run the Scraper (`src/telegram_scraper.py`):**
    This script collects raw messages, metadata, and images from the Telegram channels specified in `config/config.py`.
    Run from the project root:

    ```bash
    python src/telegram_scraper.py
    ```

    *Note: You will be prompted to enter a Telegram verification code for the first run.*
    **Output:** `data/raw/telegram_data.csv` and images in `photos/`.

2.  **Perform Initial Data Ingestion EDA (`notebooks/data_ingestion_eda.ipynb`):**
    Explore the characteristics of the raw scraped data (e.g., missing values, distribution of views/reactions, presence of images).
    Run from the project root:

    ```bash
    jupyter notebook notebooks/data_ingestion_eda.ipynb
    ```

    **Insights:**
    * Approximately 46% of messages have missing text, indicating the necessity for OCR on images.
    * 88% of messages contain images, highlighting the importance of image analysis.
    * High-engagement messages (top quartile) have 27+ reactions.

3.  **Run the Preprocessor (`src/preprocessor.py`):**
    This script cleans and normalizes the raw text data by:
    * Normalizing Amharic character variations.
    * Strictly removing emojis and pictorial symbols (without converting them to text).
    * Removing URLs and hashtags.
    * Standardizing currency expressions (e.g., "1500ብር" to "1500 ETB").
    * Retaining Telegram usernames and phone numbers.
    * Removing extra spaces and cleaning miscellaneous characters.
    Run from the project root:

    ```bash
    python src/preprocessor.py
    ```

    **Output:** `data/processed/clean_telegram_data.csv`.

4.  **Perform Preprocessing EDA (`notebooks/data_preprocessing_eda.ipynb`):**
    Analyze the characteristics of the cleaned text data, such as text length distribution and common words.
    Verify the successful removal of unwanted characters and the retention of critical entities (usernames, phone numbers).
    Run from the project root:

    ```bash
    jupyter notebook notebooks/data_preprocessing_eda.ipynb
    ```

    **Insights:**
    * Confirmed loading and basic characteristics of `clean_telegram_data.csv`.
    * Analyzed distribution of preprocessed text lengths and common words.
    * Verified retention of Telegram usernames and phone numbers.
    * Identified that ~46% of `preprocessed_text` entries are empty (corresponding to messages that were originally only emojis/images/etc.).

5.  **Run Unit Tests (`tests/test_preprocessor.py`):**
    Verify the correctness of the `preprocessor.py` functions.
    Run from the project root:

    ```bash
    pytest tests/test_preprocessor.py
    ```

## 🎯 3. Named Entity Recognition (NER) Pipeline

This section details the steps for labeling data and fine-tuning an NER model to extract key business entities.

### 3.1. Data Labeling (Task 2)

The cleaned text data is converted into a CoNLL-like format, suitable for Named Entity Recognition (NER) model training. This step involves applying rule-based labeling to identify entities such as product names, prices, locations, contact information, and delivery details.

**Script:** `src/data_labeler.py`
**Execution:**

```bash
python src/data_labeler.py
```

**Output:** `data/labeled/telegram_ner_data_rule_based.conll`
**Process:**

  * Reads `clean_telegram_data.csv`.
  * Applies a set of refined regex patterns to identify and extract entities.
  * Handles overlap resolution by prioritizing certain entity types and longer matches.
  * Converts the identified entities into the CoNLL format (Token t Tag), ensuring consistency for model training.
    **Status:** Completed. The script successfully generated the labeled `.conll` file.

### 3.2. Model Fine-tuning (Task 3)

A pre-trained multilingual transformer model (`Davlan/afro-xlmr-large`) is fine-tuned on the labeled Amharic NER dataset to accurately extract entities from new Telegram messages.

**Script:** `src/model_finetuner.py`
**Execution:**

```bash
python src/model_finetuner.py
```

**Output:** The fine-tuned model and its tokenizer are saved to `models/afro_xlmr_ner_fine_tuned/`.
**Process:**

  * **Data Loading & Splitting:** Loads the CoNLL data, parses it into sentences, and splits it into 80% training, 10% validation, and 10% test sets. Stratification is attempted to maintain class distribution, but automatically disabled for robustness with small sample sizes or imbalanced classes.
  * **Tokenization & Label Alignment:** Uses the `afro-xlmr-large` tokenizer to convert words into subword tokens and aligns the word-level NER labels to these subwords, correctly handling B-, I-, L-, U-, and O tags for sequence tagging.
  * **Model Initialization:** Loads `Davlan/afro-xlmr-large` for token classification, configuring its output layer for the defined NER labels (PRODUCT, PRICE, LOC, CONTACT, DELIVERY).
  * **Training:** Fine-tunes the model for 5 epochs using a batch size of 8, with evaluation performed at each epoch.
  * **Evaluation:** Calculates Precision, Recall, and F1-score on the validation and test sets to assess model performance.
    **Status:** Completed. The model was successfully fine-tuned and saved.

### Initial Model Performance (`afro-xlmr-large` on Test Set):

| Entity Type | Precision | Recall | F1-Score | Support |
| :---------- | :-------- | :----- | :------- | :------ |
| CONTACT     | 0.00      | 0.00   | 0.00     | 1       |
| DELIVERY    | 0.00      | 0.00   | 0.00     | 0       |
| LOC         | 0.10      | 0.05   | 0.07     | 55      |
| PRICE       | 0.01      | 0.06   | 0.01     | 16      |
| PRODUCT     | 0.02      | 0.25   | 0.03     | 4       |
| **micro avg** | **0.02** | **0.07** | **0.03** | **76** |
| **macro avg** | **0.03** | **0.07** | **0.02** | **76** |
| **weighted avg** | **0.08** | **0.07** | **0.06** | **76** |

**Summary:** The initial performance is very low across all entity types, with F1-scores close to zero. This is primarily attributed to the small training dataset (only 40 sentences for training). Transformer models require significantly more labeled data to learn robust patterns for NER. Future improvements will focus on expanding the dataset and potentially exploring data augmentation techniques.

## 🎯 4. Model Comparison & Selection (Task 4) 

This phase involves fine-tuning additional multilingual models to compare their performance against `afro-xlmr-large` on the Amharic NER task, focusing on accuracy and efficiency.

* **Objective:** Fine-tune `DistilBERT` and compare its performance with `afro-xlmr-large`.
* **Script:** `src/distilbert_finetuner.py`
* **Output:** The fine-tuned `DistilBERT` model and its tokenizer are saved to `models/distilbert_ner_fine_tuned/`.
* **Process:**
    * **Model:** `distilbert-base-multilingual-cased` was used for fine-tuning.
    * **Training:** Similar training parameters as `afro-xlmr-large` (5 epochs, batch size 8).
    * **Evaluation:** Precision, Recall, and F1-score were calculated on the test set.
* **Status:** **Completed**. The `DistilBERT` model was successfully fine-tuned and saved.

* **Model Performance Comparison (on Test Set):**

    | Metric        | `afro-xlmr-large` | `DistilBERT` |
    | :------------ | :---------------- | :----------- |
    | Eval Loss     | 2.845             | 2.960        |
    | Precision     | 0.010             | 0.055        |
    | Recall        | 0.039             | 0.132        |
    | F1-Score      | **0.016** | **0.078** |
    | Train Runtime | ~48 minutes       | **~3.7 minutes** |

    **Summary:**
    `DistilBERT` demonstrated a notably better F1-score (0.078 vs. 0.016) and significantly faster training time (~3.7 minutes vs. ~48 minutes) compared to `afro-xlmr-large` on this dataset. Despite the improvements, overall performance for both models remains low, largely due to the very limited size of the labeled dataset. Further data augmentation or more extensive labeling is crucial for achieving practical performance.

---

### Future Enhancements (Tasks 5 & 6)

* **Model Interpretability (Task 5):** Implement SHAP and LIME to explain model predictions, especially for difficult cases.
* **FinTech Vendor Scorecard for Micro-Lending (Task 6):** Develop an analytics engine to combine extracted NER entities with Telegram post metadata (views, timestamps) to calculate key vendor performance metrics (posting frequency, average views per post, average price point) and derive a "Lending Score."
