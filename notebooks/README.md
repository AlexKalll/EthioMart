# EthioMart: Jupyter Notebooks

This directory contains Jupyter notebooks used for exploratory data analysis (EDA), data preprocessing, model experimentation, and results visualization for the EthioMart project.

## Notebooks Overview:

### `data_ingestion_eda.ipynb`:

  * **Purpose:** To perform initial Exploratory Data Analysis (EDA) on the raw Telegram data collected by `src/telegram_scraper.py`.
  * **Key Analyses:**
      * Loading and inspecting the raw `telegram_data.csv`.
      * Checking for missing values in text and image paths.
      * Analyzing the distribution of views and `reactions_count`.
      * Identifying the proportion of messages with images versus text.
  * **Insights Gained:** Highlighted the significant number of messages with missing text (requiring OCR) and the prevalence of images (emphasizing visual data importance).

### `data_preprocessing_eda.ipynb`:

  * **Purpose:** To perform EDA on the cleaned and preprocessed Telegram data generated by `src/preprocessor.py`.
  * **Key Analyses:**
      * Loading and inspecting the `clean_telegram_data.csv`.
      * Analyzing the distribution of `preprocessed_text` lengths.
      * Identifying the most frequent words in the cleaned corpus.
      * Verifying the successful removal of emojis/decorative symbols and retention of key entities like Telegram usernames and phone numbers.
  * **Insights Gained:** Confirmed the effectiveness of the preprocessing steps and validated that necessary information for NER (like contacts) is preserved, while irrelevant clutter (emojis) is removed.

## How to Use These Notebooks:

1.  **Ensure Dependencies are Installed:** Make sure you have activated your project's virtual environment and installed all packages from `requirements.txt`.

2.  **Run Scraper and Preprocessor:** Before running these EDA notebooks, ensure you have run `src/telegram_scraper.py` and `src/preprocessor.py` to generate the raw and cleaned CSV files.

3.  **Launch Jupyter:** Navigate to the `EthioMart/` root directory .

4.  **Open Notebooks:** From the Jupyter interface, navigate to the `notebooks/` directory and open `data_ingestion_eda.ipynb` or `data_preprocessing_eda.ipynb`.

5.  **Execute Cells:** Run the cells sequentially to see the data loading, analysis, and visualizations.