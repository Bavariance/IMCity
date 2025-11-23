import os
import logging
from dotenv import load_dotenv

from bavariance_bot import BavarianceBot
from settings import ROOT_DIR


# --- LOGGING CONFIGURATION ---
LOG_TO_CONSOLE = True      # Set to False to hide output from terminal
LOG_TO_FILE = True         # Set to True to save output to 'bot_execution.log'
LOG_LEVEL = logging.INFO   # Use logging.DEBUG for detailed math/logic, logging.INFO for trades only


def setup_logging():
    """Configures logging based on global settings."""
    handlers = []
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if LOG_TO_FILE:
        file_handler = logging.FileHandler("bot_execution.log")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=LOG_LEVEL,
        handlers=handlers,
        force=True # Ensures we overwrite any default library configs
    )


def main():
    setup_logging()
    logging.info("Initializing Bot...")

    load_dotenv(dotenv_path=ROOT_DIR / '.env')

    TEST_EXCHANGE = os.getenv("TEST_EXCHANGE")
    REAL_EXCHANGE = os.getenv("REAL_EXCHANGE")
    username = os.getenv("IMC_USERNAME")
    password = os.getenv("IMC_PASSWORD")

    # CTRL + C to stop the bot
    bot = BavarianceBot(REAL_EXCHANGE, username, password)
    bot.start() 


if __name__ == "__main__":
    main()
