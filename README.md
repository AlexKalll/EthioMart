# EthioMart | Amharic E-commerce Data Extractor

## 📘 Project Overview

EthioMart aims to become the primary hub for Telegram-based e-commerce in Ethiopia by centralizing real-time data from various channels. This project focuses on building an Amharic Named Entity Recognition (NER) system to extract key business entities (product names, prices, locations, contacts) from Telegram messages and images. The extracted data will populate EthioMart's centralized database, enabling a seamless experience for customers and informing FinTech initiatives like vendor loan assessments.

This project leverages Python, Telegram API, and data science tools to build a robust data pipeline, from scraping and preprocessing to model fine-tuning and interpretability.

## 🏗️ 1. Project Setup and Data Collection (Task 1)

**Branch created:** `task1` for initial project setup and data scraping.

**Deliverables:**

  * GitHub code for Task 1 (data ingestion and preprocessing).
  * Data summary (1-2 pages) covering data preparation and labeling steps.

### Repo/Project Structure

```
EthioMart/
    ├── src/
    │   ├── telegram_scraper.py # Collects raw data from Telegram channels
    │   └── preprocessor.py     # Cleans and preprocesses raw text data
    ├── config/
    │   └── config.py           # Stores configuration variables (e.g., API credentials, channel list)
    ├── data/
    │   ├── raw/                # Stores raw scraped data (e.g., telegram_data.csv)
    │   │   └── telegram_data.csv
    │   ├── processed/          # Stores cleaned and preprocessed data
    │   │   └── clean_telegram_data.csv
    │   └── labeled/            # Will store manually and semi-automatically labeled data (for Task 2)
    ├── photos/                 # Stores downloaded images from Telegram messages
    ├── notebooks/              # Jupyter notebooks for EDA, experimentation, and documentation
    │   ├── data_ingestion_eda.ipynb
    │   └── data_preprocessing_eda.ipynb
    ├── outputs/                # Stores generated plots and visualizations
    ├── reports/                # For interim and final project reports
    ├── tests/                  # Unit tests for various modules (e.g., preprocessor)
    │   └── test_preprocessor.py
    ├── .github/workflows/      # CI/CD pipelines (e.g., for DVC and code quality)
    ├── .env                    # Environment variables (e.g., Telegram API keys)
    ├── requirements.txt        # Python package dependencies
    ├── .gitignore              # Files/directories to ignore in Git
    ├── README.md               # Project overview and setup instructions
    └── DVC.md                  # Documentation for Data Version Control (DVC) setup (Future)
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

    ```
    TELEGRAM_API_ID=your_api_id
    TELEGRAM_API_HASH=your_api_hash
    TELEGRAM_PHONE_NUMBER=your_phone_number
    ```

    Obtain `API_ID` and `API_HASH` from [my.telegram.org](https://my.telegram.org/).

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
    *Output: `data/raw/telegram_data.csv` and images in `photos/`.*

2.  **Perform Initial Data Ingestion EDA (`notebooks/data_ingestion_eda.ipynb`):**
    Explore the characteristics of the raw scraped data (e.g., missing values, distribution of views/reactions, presence of images).
    Run from the project root:

    ```bash
    jupyter notebook notebooks/data_ingestion_eda.ipynb
    ```

    *Insights:*

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

    <!-- end list -->

    ```bash
    python src/preprocessor.py
    ```

    *Output: `data/processed/clean_telegram_data.csv`.*

4.  **Perform Preprocessing EDA (`notebooks/data_preprocessing_eda.ipynb`):**
    Analyze the characteristics of the cleaned text data, such as text length distribution and common words.
    Verify the successful removal of unwanted characters and the retention of critical entities (usernames, phone numbers).
    Run from the project root:

    ```bash
    jupyter notebook notebooks/data_preprocessing_eda.ipynb
    ```

    *Insights:*

      * Confirmed loading and basic characteristics of `clean_telegram_data.csv`.
      * Analyzed distribution of preprocessed text lengths and common words.
      * Verified retention of Telegram usernames and phone numbers.
      * Identified that \~46% of `preprocessed_text` entries are empty (corresponding to messages that were originally only emojis/images/etc.).

5.  **Run Unit Tests (`tests/test_preprocessor.py`):**
    Verify the correctness of the `preprocessor.py` functions.
    Run from the project root:

    ```bash
    pytest tests/test_preprocessor.py
    ```

## 🎯 3. Next Steps (Task 2 onwards)

The next phase will focus on preparing the data for the Named Entity Recognition (NER) task.

  * **Data Labeling:** Convert cleaned text and existing labeled data into the CoNLL format. This will involve:
      * Defining and applying labels for product, price, location, contact, and delivery entities.
      * Addressing any remaining text overlapping issues (e.g., "ዋጋ ስልክ አድራሻ" or "price contact") by careful tokenization and labeling strategy.
  * **Data Splitting:** Divide the labeled data into training, validation, and test sets for model development.
  * **Data Versioning:** Set up DVC to version `telegram_data.csv` and large image datasets in `photos/`.
  * **Model Fine-Tuning:** Plan for fine-tuning Amharic LLM models for NER.
  * **Model Interpretability:** Integrate tools like SHAP/LIME for understanding model predictions.