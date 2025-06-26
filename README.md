# EthioMart | Amharic E-commerce Data Extractor

## 📘 Project Overview

EthioMart aims to become the primary hub for Telegram-based e-commerce in Ethiopia by centralizing real-time data from various channels. This project focuses on building an Amharic Named Entity Recognition (NER) system to extract key business entities (product names, prices, locations, contacts) from Telegram messages and images. The extracted data will populate EthioMart's centralized database, enabling a seamless experience for customers and informing FinTech initiatives like vendor loan assessments.

This project leverages Python, Telegram API, and data science tools to build a robust data pipeline, from scraping and preprocessing to model fine-tuning and interpretability.

## 🏗️ 1. Project Setup, Data Collection and Preprocessing

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
├── outputs/                # Stores generated scorecard, plots and visualizations
│   └── plots/
│   └── vendor_scorecard.csv
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

### 🚀 2. Usage and Data Pipeline Steps

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
    This script collects raw messages, metadata, and images from the Telegram channels specified in `config/config.py`. The channels where we scrape the messages are `'@ZemenExpress', '@ethio_brand_collection', '@Leyueqa', '@Fashiontera', and '@marakibrand'`.
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

### 3.1. Data Labeling 

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

## 🎯3. Model Fine-tuning 

A pre-trained multilingual transformer model (`Davlan/afro-xlmr-large`) is fine-tuned on the labeled Amharic NER dataset to accurately extract entities from new Telegram messages.

**Script:** `src/model_finetuner.py`, excute in the root dir as:

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

#### Initial Model Performance (`afro-xlmr-large` on Test Set):

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

## 🎯 4. Model Comparison & Selection 

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

## 🎯5. Model Interpretability 

This phase explores how the fine-tuned NER model makes its predictions using interpretability tools.

**Objective:** Implement SHAP and conceptually outline LIME to understand model decision-making.
**Notebook:** `notebooks/model_interpretability.ipynb`
**Process:**
* The best-performing model (DistilBERT) was loaded for inference.
* SHAP (SHapley Additive exPlanations) was implemented to show word-level contributions to entity predictions for specific examples.
* LIME (Local Interpretable Model-agnostic Explanations) was conceptually discussed due to its complexity for token-level NER.
**Status:** Implemented (with known issues). While SHAP explanation code is present, a `TypeError` prevented full execution within the given time constraints, and the overall low model performance limits the depth of meaningful interpretation.

---

## 🎯6 FinTech Vendor Scorecard for Micro-Lending

This task focuses on combining NER-extracted entities with Telegram post metadata to build a vendor analytics engine and generate a "Lending Score" for potential micro-lending candidates.

**Objective:** Develop a script to calculate key vendor performance metrics and a composite lending score.
**Script:** `src/vendor_scorecard_engine.py`
**Output:** A summary table of vendor metrics and a CSV file saved to `outputs/vendor_scorecard.csv`.
**Process:**
* Loads `clean_telegram_data.csv`.
* Utilizes the fine-tuned DistilBERT NER model to extract product, price, location, contact, and delivery entities from all preprocessed messages.
* Calculates Posting Frequency, Average Views per Post, identifies the Top Performing Post (including its extracted product and price), and computes the Average Price Point for each vendor channel.
* Derives a `Lending_Score` based on a weighted combination of Average Views per Post and Posting Frequency.
**Status:** Completed. The vendor scorecard was successfully generated and saved.

### Vendor Scorecard Sample Output:

| Vendor_Channel | Posting_Frequency_per_Week | Average_Views_per_Post | Top_Product | Top_Price | Average_Price_Point_ETB | Lending_Score |
| :---------------- | :------------------------ | :--------------------- | :---------- | :-------- | :---------------------- | :------------ |
| Zemen Express®    | 42.424242                 | 5417.891               | None        | None      | 1.664871e+07            | 2730.157621   |
| EthioBrand®       | 10.494753                 | 39753.976              | ##ge        | SD Size   | 1.624311e+13            | 19882.235376  |
| ልዩ እቃ          | 41.666667                 | 26020.603              | LeM          |ዘመናዊ አዲስ በኤሌክትሪክ የሚሰራ  | 1.534902e+10            | 13031.134833  |
| Fashion tera      | 5.359877                  | 9385.297               | 2           | ፋሽን ተራ    | 3.179614e+09            | 4695.328439   |
| ማራኪ ცЯﾑŋの™    | 21.671827                 | 11434.001              | None        | None      | 3.293501e+08            | 5727.836413   |

#### Observations & Limitations:

* The `Top_Product` and `Top_Price` fields frequently appear as `None` or contain incorrect/partial extractions (e.g., `##ge`, `SD Size`, `ፋሽን ተራ`). This is a direct consequence of the low F1-scores of the underlying NER model (Task 4), which struggled with accurate entity extraction on the small labeled dataset.
* The `Average_Price_Point_ETB` values are extremely high (e.g., in the billions/trillions). This indicates an issue with the price extraction and numeric conversion logic, likely due to the NER model misidentifying non-price numbers as prices or improper parsing of extracted price strings from the NER output.
* The `Lending_Score` currently reflects engagement metrics more reliably than business profile metrics (product/price) due to the NER model's limitations.

**Crucial Next Step for Improvement:** The most significant enhancement for the FinTech scorecard is to expand the labeled dataset for the NER model. A more accurate NER model will directly improve the quality of `Top_Product`, `Top_Price`, and `Average_Price_Point_ETB`, making the `Lending_Score` much more robust and actionable for EthioMart. Refining the price parsing logic within `vendor_scorecard_engine.py` is also necessary.


