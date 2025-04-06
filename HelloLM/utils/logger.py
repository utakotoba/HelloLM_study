import sys
import datetime
from loguru import logger
from HelloLM.utils.path import ensure_directory


def generate_log_key():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return timestamp


def setup_logger(disable_log_to_file: bool, log_file_path: str, log_file_split: bool):
    # disable builtin logger style
    logger.remove()
    # message logger to stdout and stderr
    logger.add(
        sys.stdout,
        format="<d><g>{time:YYYY-MM-DD HH:mm:ss}</g></d> | <d><g>{file.name}:{function}:{line}</g></d> | <g>{level}</g> | {message}",
        colorize=True,
        filter=lambda record: record["level"].name == "INFO",
    )
    logger.add(
        sys.stderr,
        format="<n><e>{time:YYYY-MM-DD HH:mm:ss}</e></n> | <n><e>{file.name}:{function}:{line}</e></n> | <e>{level}</e> | <e>{message}</e>",
        colorize=True,
        filter=lambda record: record["level"].name == "DEBUG",
    )
    logger.add(
        sys.stderr,
        format="<n><y>{time:YYYY-MM-DD HH:mm:ss}</y></n> | <n><y>{file.name}:{function}:{line}</y></n> | <y>{level}</y> | <y>{message}</y>",
        colorize=True,
        filter=lambda record: record["level"].name == "WARNING",
    )
    logger.add(
        sys.stderr,
        format="<b><r>{time:YYYY-MM-DD HH:mm:ss}</r></b> | <b><r>{file.name}:{function}:{line}</r></b> | <r>{level}</r> | <r>{message}</r>",
        colorize=True,
        filter=lambda record: record["level"].name == "ERROR",
    )

    # setup file logging
    if not disable_log_to_file:
        ensure_directory(log_file_path)
        key = generate_log_key()
        if log_file_split:
            resolved_path = f"{log_file_path}/{key}"
            ensure_directory(resolved_path)
            logger.add(
                f"{resolved_path}/info_{key}.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {file.name}:{function}:{line} | {level} | {message}",
                filter=lambda record: record["level"].name == "INFO",
            )
            logger.add(
                f"{resolved_path}/debug_warning_error_{key}.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {file.name}:{function}:{line} | {level} | {message}",
                filter=lambda record: record["level"].name
                in ["DEBUG", "WARNING", "ERROR"],
            )
        else:
            logger.add(
                f"{log_file_path}/{key}.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {file.name}:{function}:{line} | {level} | {message}",
            )

    # report
    logger.info("Logger successfully initialized!")
