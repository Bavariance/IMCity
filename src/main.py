from bavariance_bot import BavarianceBot


TEST_EXCHANGE = "" # TODO
REAL_EXCHANGE = "" # TODO


def main():
    # TODO: Load from config/env file
    username = "myuser"  # TODO
    password = "secret"  # TODO

    try:
        bot = BavarianceBot(TEST_EXCHANGE, username, password)
        # bot.start()

        # while True:
        #     pass
    except KeyboardInterrupt as e:
        bot.stop()


if __name__ == "__main__":
    main()
