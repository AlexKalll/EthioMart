# src/telegram_scraper.py
import asyncio
import csv
from datetime import datetime
from pathlib import Path
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto
import logging
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add parent directory to path
# Import configurations from your central config.py
from config.config import TelegramConfig, TARGET_CHANNELS, DATA_DIR, PHOTOS_DIR

# Configure logging for the scraper
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def get_reaction_count(message):
    """Count total reactions for a message"""
    if not message.reactions:
        return 0
    # Sum up counts from all reaction types (e.g., likes, dislikes, love etc.)
    return sum(reaction.count for reaction in message.reactions.results)

async def scrape_all_channels(client):
    """Scrape all channels and save data to CSV and download images to photos directory"""
    csv_path = DATA_DIR / "telegram_data.csv"
    
    # Ensure photos directory exists (already handled in config.py, but good to double check)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'channel_title', 
            'message_id', 
            'date', 
            'text', 
            'views', 
            'reactions_count', 
            'image_path' # Relative path like "../photos/@channel_name_123.jpg"
        ])
        
        logging.info(f"Starting scraping of {len(TARGET_CHANNELS)} channels...")
        for channel_username in TARGET_CHANNELS: 
            logging.info(f"Scraping {channel_username}...")
            try:
                channel_entity = await client.get_entity(channel_username)
                
                # Limit to 1000 messages per channel as a default for testing
                async for message in client.iter_messages(channel_entity, limit=1000): 
                    try:
                        # Format date as YYYY-MM-DD
                        date_only = message.date.date().isoformat()
                        
                        # Get reaction count
                        reactions_count = await get_reaction_count(message)
                        
                        # Handle photo media with relative path
                        image_relative_path = ''
                        if isinstance(message.media, MessageMediaPhoto):
                            # Remove '@' from channel username for filename
                            image_filename = f"{channel_username[1:]}_{message.id}.jpg" 
                            full_image_path = PHOTOS_DIR / image_filename
                            
                            # Download image
                            await client.download_media(message.media, file=full_image_path)
                            
                            # Calculate relative path from the perspective of data/raw/telegram_data.csv
                            # If telegram_data.csv is in data/raw/, and images in photos/, 
                            # then to reference 'photos/image.jpg' from data/raw/ we need '../photos/image.jpg'
                            # Path.relative_to provides this, but simpler to construct explicitly here
                            image_relative_path = str(Path("../photos") / image_filename).replace("\\", "/") # Ensure Unix-like path separators
                        
                        # Write row to CSV
                        writer.writerow([
                            channel_entity.title,
                            message.id,
                            date_only,
                            message.text or '', # Use empty string if message.text is None
                            message.views or 0, # Use 0 if message.views is None
                            reactions_count,
                            image_relative_path
                        ])
                    except Exception as e:
                        logging.error(f"Error processing message {message.id} from {channel_username}: {str(e)}")
                        
            except Exception as e:
                logging.error(f"Failed to scrape {channel_username}: {str(e)}")
    
    logging.info(f"Scraping complete. Data saved to: {csv_path}")
    logging.info(f"Images saved to: {PHOTOS_DIR}")

async def main():
    """Main function to run the scraper"""
    # Validate Telegram credentials before starting
    TelegramConfig.validate()
    
    # Initialize Telegram client
    client = TelegramClient(
        'ethiomart_session', # Session name
        TelegramConfig.API_ID,
        TelegramConfig.API_HASH
    )
    
    async with client: # Use async context manager for proper client lifecycle
        # Authenticate if not already authorized
        if not await client.is_user_authorized():
            logging.info("Authorizing Telegram client...")
            await client.send_code_request(TelegramConfig.PHONE)
            try:
                # Prompt user for the code in the terminal
                user_code = input('Enter the verification code sent to your Telegram app: ')
                await client.sign_in(TelegramConfig.PHONE, user_code)
                logging.info("Telegram client authorized successfully.")
            except Exception as e:
                logging.error(f"Failed to sign in: {e}")
                return # Exit if sign-in fails
        
        await scrape_all_channels(client)

if __name__ == '__main__':
    # Run the main asynchronous function
    asyncio.run(main())