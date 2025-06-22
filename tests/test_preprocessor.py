# EthioMart/tests/test_preprocessor.py

import pytest
import pandas as pd
from pathlib import Path
import sys
import shutil

# Add the project root to sys.path to allow importing from src and config
project_root = Path(__file__).resolve().parent.parent # Points to EthioMart/
sys.path.insert(0, str(project_root)) # Insert at the beginning to prioritize

from src.preprocessor import preprocess_amharic, validate_csv # Import functions to test
from config.config import DATA_DIR # Import DATA_DIR from your config

# Define temporary directories for test output files
TEST_OUTPUT_DIR = Path(__file__).parent / "temp_preprocessor_test_data"
TEST_PROCESSED_DIR = TEST_OUTPUT_DIR / "data" / "processed"
TEST_CLEANED_CSV = TEST_PROCESSED_DIR / "clean_telegram_data.csv"

# --- Pytest Fixture for Test Setup/Teardown ---
@pytest.fixture(scope="module", autouse=True)
def setup_teardown_test_environment():
    """
    Sets up and tears down the test environment for preprocessor tests.
    Ensures a clean state for each test run.
    """
    # Setup: Create necessary test directories
    TEST_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    yield # This runs the tests

    # Teardown: Clean up test directories after all tests in this module are done
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)

# --- Tests for preprocess_amharic function ---

def test_preprocess_amharic_emoji_removal():
    """Test emoji and symbol removal."""
    text = "Hello 😊 world! 💥 Telegram post 🚀"
    expected = "Hello world! Telegram post"
    assert preprocess_amharic(text) == expected

    text_amharic = "እንኳን ደህና መጡ 🎉 ወደዚህ ቻናል 👍"
    expected_amharic = "እንኳን ደህና መጡ ወደዚህ ቻናል"
    assert preprocess_amharic(text_amharic) == expected_amharic

def test_preprocess_amharic_telegram_patterns_removal():
    """Test removal of Telegram-specific decorative patterns."""
    text = "📌Product Info📍🔥New Arrival✨〰️〰️〰️Test---Line###"
    expected = "Product Info New Arrival Test Line" # Hash tags are removed
    assert preprocess_amharic(text) == expected

    text = "አድራሻ: አዲስ አበባ ,ጦር ሀይሎች ድሪም ታወር 2ተኛ ፎቅ" # Example with comma and general Amharic text
    expected = "አድራሻ አዲስ አበባ , ጦር ሀይሎች ድሪም ታወር 2ተኛ ፎቅ" # Comma should remain
    assert preprocess_amharic(text) == expected

def test_preprocess_amharic_url_mention_hashtag_removal():
    """Test URL, mention, and hashtag removal."""
    text = "Check out this link: https://example.com and follow @user #awesome"
    expected = "Check out this link: and follow" # Hashtag and mention removed
    assert preprocess_amharic(text) == expected

def test_preprocess_amharic_currency_standardization():
    """Test currency standardization."""
    text = "ዋጋ፦ 💲🏷 2700 ብር ✅ and 1,500Br 💵 also 200 ETB"
    expected = "ዋጋ፦ 2700 ETB and 1500 ETB also 200 ETB"
    assert preprocess_amharic(text) == expected

def test_preprocess_amharic_phone_number_standardization():
    """Test phone number standardization."""
    text = "Call us at +251911223344 or 0987654321 and 900112233"
    expected = "Call us at <PHONE_NUMBER> or <PHONE_NUMBER> and <PHONE_NUMBER>"
    assert preprocess_amharic(text) == expected

def test_preprocess_amharic_multiple_spaces_and_strip():
    """Test multiple space replacement and stripping."""
    text = "  Hello   World   "
    expected = "Hello World"
    assert preprocess_amharic(text) == expected

def test_preprocess_amharic_non_amharic_non_english_removal():
    """Test removal of unwanted non-alphanumeric characters (excluding Amharic, English, digits, punctuation)."""
    text = "This is a test with some weird chars: £€¥[]{}()"
    expected = "This is a test with some weird chars: " # Punctuation like :,.!? is kept based on regex
    assert preprocess_amharic(text) == expected

    text_amharic_mixed = "ይህ አንድ የሙከራ ጽሁፍ ነው። 123!"
    expected_amharic_mixed = "ይህ አንድ የሙከራ ጽሁፍ ነው። 123!"
    assert preprocess_amharic(text_amharic_mixed) == expected_amharic_mixed

def test_preprocess_amharic_empty_and_none_input():
    """Test handling of empty string and None input."""
    assert preprocess_amharic("") == ""
    assert preprocess_amharic(None) == ""
    assert preprocess_amharic(123) == "" # Test non-string input

def test_preprocess_amharic_complex_case():
    """Test a more complex example combining multiple rules."""
    complex_text = "😊Hello World!💥 Call 📲 0912345678. Price: 1,200 ብር. Visit us: https://example.com #best"
    expected_cleaned = "Hello World! Call <PHONE_NUMBER> . Price: 1200 ETB. Visit us:"
    assert preprocess_amharic(complex_text) == expected_cleaned


# --- Tests for validate_csv function ---

def test_validate_csv_success():
    """Test validate_csv with a valid DataFrame."""
    # Create a dummy CSV file that is valid
    valid_data = {
        'channel_title': ['Ch1', 'Ch2'],
        'message_id': [1, 2],
        'date': ['2023-01-01', '2023-01-02'],
        'text': ['raw text 1', 'raw text 2'],
        'views': [10, 20],
        'reactions_count': [1, 2],
        'image_path': ['', 'path/to/img.jpg'],
        'preprocessed_text': ['clean text 1', 'clean text 2']
    }
    df_valid = pd.DataFrame(valid_data)
    
    # Ensure the directory exists for the test CSV
    TEST_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_valid.to_csv(TEST_CLEANED_CSV, index=False, encoding='utf-8')

    # No exception means success
    validate_csv(str(TEST_CLEANED_CSV)) # validate_csv expects a string path

def test_validate_csv_missing_file():
    """Test validate_csv with a non-existent file."""
    with pytest.raises(FileNotFoundError, match="Processed CSV file not found"):
        validate_csv(str(TEST_PROCESSED_DIR / "non_existent.csv"))

def test_validate_csv_missing_columns():
    """Test validate_csv with missing columns."""
    invalid_data = {
        'message_id': [1, 2],
        'text': ['raw text 1', 'raw text 2'],
        'preprocessed_text': ['clean text 1', 'clean text 2']
    }
    df_invalid = pd.DataFrame(invalid_data)
    df_invalid.to_csv(TEST_CLEANED_CSV, index=False, encoding='utf-8')

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_csv(str(TEST_CLEANED_CSV))

def test_validate_csv_null_message_id():
    """Test validate_csv with NULL values in message_id."""
    null_id_data = {
        'channel_title': ['Ch1', 'Ch2'],
        'message_id': [1, None], # NULL value
        'date': ['2023-01-01', '2023-01-02'],
        'text': ['raw text 1', 'raw text 2'],
        'views': [10, 20],
        'reactions_count': [1, 2],
        'image_path': ['', 'path/to/img.jpg'],
        'preprocessed_text': ['clean text 1', 'clean text 2']
    }
    df_null_id = pd.DataFrame(null_id_data)
    df_null_id.to_csv(TEST_CLEANED_CSV, index=False, encoding='utf-8')

    with pytest.raises(ValueError, match="NULL values in 'message_id' column"):
        validate_csv(str(TEST_CLEANED_CSV))

def test_validate_csv_empty_preprocessed_text_warning(caplog):
    """Test validate_csv with original text becoming empty preprocessed text, which should log a warning."""
    # This test uses caplog to capture logging messages
    with caplog.at_level(logging.WARNING):
        data = {
            'channel_title': ['Ch1'],
            'message_id': [1],
            'date': ['2023-01-01'],
            'text': ['Just emojis 😊💥👍'], # This will be cleaned to an empty string
            'views': [10],
            'reactions_count': [1],
            'image_path': [''],
            'preprocessed_text': [''] # Simulate it becoming empty
        }
        df_empty_preprocessed = pd.DataFrame(data)
        df_empty_preprocessed.to_csv(TEST_CLEANED_CSV, index=False, encoding='utf-8')

        validate_csv(str(TEST_CLEANED_CSV))
        assert "Some messages that originally had text resulted in empty or null preprocessed_text." in caplog.text

