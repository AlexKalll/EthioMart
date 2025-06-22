# config/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TelegramConfig:
    API_ID = int(os.getenv('TELEGRAM_API_ID'))
    API_HASH = os.getenv('TELEGRAM_API_HASH')
    PHONE = os.getenv('TELEGRAM_PHONE_NUMBER')
    
    @classmethod
    def validate(cls):
        if not all([cls.API_ID, cls.API_HASH, cls.PHONE]):
            raise ValueError("Missing Telegram credentials in .env file")

# Channel configuration
TARGET_CHANNELS = [
    '@ZemenExpress',
    '@ethio_brand_collection',
    '@Leyueqa',
    '@Fashiontera',
    '@marakibrand'
]

# Path configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
PHOTOS_DIR = BASE_DIR / 'photos'

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)