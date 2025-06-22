# EthioMart/tests/test_telegram_scraper.py

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import os
import csv

# We need to add the project root to sys.path so we can import from src and config
# This ensures the test can find your scraper and config files.
import sys
project_root = Path(__file__).resolve().parent.parent # Points to EthioMart/
sys.path.insert(0, str(project_root)) # Insert at the beginning to prioritize

# Import the main scraping function and config details
# We will mock the actual TelegramClient and its methods
from src.telegram_scraper import scrape_all_channels, get_reaction_count
from config.config import DATA_DIR, PHOTOS_DIR, TARGET_CHANNELS, TelegramConfig

# Define a temporary directory for test outputs
# This ensures tests are isolated and don't interfere with actual data
TEST_OUTPUT_DIR = project_root / "tests" / "temp_test_data"
TEST_DATA_DIR = TEST_OUTPUT_DIR / "data" / "raw"
TEST_PHOTOS_DIR = TEST_OUTPUT_DIR / "photos"
TEST_CSV_PATH = TEST_DATA_DIR / "telegram_data.csv"

# --- Helper for cleaning up test data ---
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """
    Fixture to create and clean up test directories before and after each test.
    This ensures a clean slate for every test run.
    """
    # Setup: Create necessary test directories
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEST_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    yield # This runs the test function

    # Teardown: Clean up test directories
    import shutil
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)

# --- Mock Message Class for Simulation ---
class MockMessage:
    """
    A mock class to simulate a Telegram message object.
    We'll only include attributes our scraper tries to access.
    """
    def __init__(self, id, text, date, views, reactions_results=None, media_type=None):
        self.id = id
        self.text = text
        self.date = date # Should be a datetime object for date().isoformat()
        self.views = views
        self.media = None
        if media_type == 'photo':
            self.media = AsyncMock() # Simulate a photo media object

        # Mock reactions object if provided
        self.reactions = None
        if reactions_results:
            # Create mock reaction results with counts
            class MockReactionResult:
                def __init__(self, count):
                    self.count = count
            self.reactions = AsyncMock()
            self.reactions.results = [MockReactionResult(c) for c in reactions_results]
        else:
            self.reactions = AsyncMock() # Empty reactions
            self.reactions.results = []

    async def download_media(self, file=None):
        """Simulate media download."""
        if file:
            # Create a dummy file to simulate download
            Path(file).touch()

# --- Mock Channel Entity Class for Simulation ---
class MockChannelEntity:
    """A mock class to simulate a Telegram channel entity object."""
    def __init__(self, title, username):
        self.title = title
        self.username = username

# --- Tests ---

@pytest.mark.asyncio
async def test_get_reaction_count():
    """Test the get_reaction_count helper function."""
    # Test with reactions
    mock_msg_with_reactions = MockMessage(id=1, text="Hello", date=datetime.now(), views=10, reactions_results=[10, 5])
    count = await get_reaction_count(mock_msg_with_reactions)
    assert count == 15

    # Test without reactions
    mock_msg_no_reactions = MockMessage(id=2, text="No reactions", date=datetime.now(), views=5)
    count = await get_reaction_count(mock_msg_no_reactions)
    assert count == 0

@pytest.mark.asyncio
@patch('src.telegram_scraper.DATA_DIR', new=TEST_DATA_DIR)
@patch('src.telegram_scraper.PHOTOS_DIR', new=TEST_PHOTOS_DIR)
@patch('src.telegram_scraper.TARGET_CHANNELS', new=['@TestChannel1', '@TestChannel2'])
@patch('src.telegram_scraper.TelegramConfig.validate', new=AsyncMock()) # Mock validation
@patch('src.telegram_scraper.TelegramClient') # Patch the TelegramClient itself
async def test_scrape_all_channels(MockTelegramClient):
    """
    Test the main scraping logic in scrape_all_channels.
    This test mocks the TelegramClient to avoid actual API calls.
    """
    # Configure the mocked TelegramClient instance
    mock_client_instance = MockTelegramClient.return_value.__aenter__.return_value # Access the client within the async with block
    mock_client_instance.is_user_authorized.return_value = True # Assume user is already authorized

    # Mock get_entity to return a mock channel
    mock_client_instance.get_entity.side_effect = [
        MockChannelEntity(title="Test Channel One", username="@TestChannel1"),
        MockChannelEntity(title="Test Channel Two", username="@TestChannel2")
    ]

    # Mock iter_messages to return a list of mock messages for each channel
    # We use a list of iterators to simulate messages from multiple channels
    mock_client_instance.iter_messages.side_effect = [
        # Messages for @TestChannel1
        [
            MockMessage(id=101, text="Text message from channel 1", date=datetime(2025, 6, 20), views=50, reactions_results=[1, 2]),
            MockMessage(id=102, text=None, date=datetime(2025, 6, 21), views=100, reactions_results=[3], media_type='photo'),
        ],
        # Messages for @TestChannel2
        [
            MockMessage(id=201, text="Message from channel 2", date=datetime(2025, 6, 22), views=20, reactions_results=[0]),
        ]
    ]

    # Run the scraping function
    await scrape_all_channels(mock_client_instance)

    # --- Assertions ---

    # 1. Check if CSV file was created and contains expected data
    assert TEST_CSV_PATH.exists()
    df = pd.read_csv(TEST_CSV_PATH)

    assert len(df) == 3 # Total messages scraped
    assert list(df.columns) == ['channel_title', 'message_id', 'date', 'text', 'views', 'reactions_count', 'image_path']

    # Verify content of the first message
    assert df.loc[0, 'channel_title'] == "Test Channel One"
    assert df.loc[0, 'message_id'] == 101
    assert df.loc[0, 'date'] == "2025-06-20"
    assert df.loc[0, 'text'] == "Text message from channel 1"
    assert df.loc[0, 'views'] == 50
    assert df.loc[0, 'reactions_count'] == 3 # 1+2
    assert df.loc[0, 'image_path'] == "" # No photo for this message

    # Verify content of the second message (with photo and no text)
    assert df.loc[1, 'channel_title'] == "Test Channel One"
    assert df.loc[1, 'message_id'] == 102
    assert df.loc[1, 'date'] == "2025-06-21"
    assert df.loc[1, 'text'] == "" # Should be empty string if None
    assert df.loc[1, 'views'] == 100
    assert df.loc[1, 'reactions_count'] == 3 # From mock reaction results
    assert df.loc[1, 'image_path'] == "../photos/TestChannel1_102.jpg"

    # Verify content of the third message
    assert df.loc[2, 'channel_title'] == "Test Channel Two"
    assert df.loc[2, 'message_id'] == 201
    assert df.loc[2, 'date'] == "2025-06-22"
    assert df.loc[2, 'text'] == "Message from channel 2"
    assert df.loc[2, 'views'] == 20
    assert df.loc[2, 'reactions_count'] == 0 # From mock reaction results
    assert df.loc[2, 'image_path'] == ""

    # 2. Check if image files were attempted to be downloaded
    # Note: Mocking download_media creates a touch() file.
    assert (TEST_PHOTOS_DIR / "TestChannel1_102.jpg").exists()
    assert not (TEST_PHOTOS_DIR / "TestChannel1_101.jpg").exists() # No photo for msg 101
    assert not (TEST_PHOTOS_DIR / "TestChannel2_201.jpg").exists() # No photo for msg 201

    # 3. Verify that TelegramClient methods were called as expected
    MockTelegramClient.assert_called_once_with('ethiomart_session', TelegramConfig.API_ID, TelegramConfig.API_HASH)
    mock_client_instance.get_entity.assert_any_call('@TestChannel1')
    mock_client_instance.get_entity.assert_any_call('@TestChannel2')
    assert mock_client_instance.iter_messages.call_count == 2 # Called once for each channel
    mock_client_instance.download_media.assert_called_once() # Called for the one message with photo
