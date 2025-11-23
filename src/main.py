import os
from dotenv import load_dotenv

from bavariance_bot import BavarianceBot
from settings import ROOT_DIR


def main():
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
