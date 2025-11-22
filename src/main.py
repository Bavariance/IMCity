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

    try:
        bot = BavarianceBot(TEST_EXCHANGE, username, password)
        # TODO: Uncomment the code below
        # bot.start()

        # while True:
        #     pass
    except KeyboardInterrupt as e:
        bot.stop()


if __name__ == "__main__":
    main()
