from argparse import ArgumentParser
from HelloLM.utils.logger import logger, setup_logger

if __name__ == "__main__":
    # resolve arguments
    parser = ArgumentParser(description="Just a simple CLI to train HelloLM (demo)")

    # log control
    logging_options = parser.add_argument_group("logging options")
    logging_options.add_argument(
        "--disable-log-to-file", help="whether to log to file", action="store_true", default=False
    )
    logging_options.add_argument(
        "--log-file-path", help="target path to save logging files", default="logs"
    )
    logging_options.add_argument(
        "--log-file-split",
        help="whether to split warning/error to another log file",
        action="store_true",
        default=False,
    )

    # parse
    args = parser.parse_args()

    # initialize logger
    setup_logger(
        disable_log_to_file=args.disable_log_to_file,
        log_file_path=args.log_file_path,
        log_file_split=args.log_file_split,
    )
