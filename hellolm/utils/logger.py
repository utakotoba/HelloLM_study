from sys import stderr
from pathlib import Path
from loguru import logger
from typing import TypedDict
from hellolm.utils.tools import ensure_directory


class LoggerConfig(TypedDict):
    log_to_file: bool
    log_path: Path


defaults: LoggerConfig = {"log_to_file": True, "log_path": Path("logs")}


# setup logging to shell and files (TensorBoard excluded)
@logger.catch
def setup_logger(
    identifier: str = "main",
    run_id: str = None,
    logger_config: LoggerConfig = defaults,
):
    # disable built-in logger handler
    logger.remove()

    # message to stderr
    logger.add(
        stderr,
        format=f"<d>{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {identifier} {{process}} | {{file.name}}:{{function}}:{{line}} | {{message}}</d>",
        colorize=True,
        filter=lambda record: record["level"].name == "TRACE",
        enqueue=True,
        level="TRACE"
    )
    logger.add(
        stderr,
        format=f"<n><e>{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {identifier} {{process}} | {{file.name}}:{{function}}:{{line}} | {{message}}</e></n>",
        colorize=True,
        filter=lambda record: record["level"].name == "DEBUG",
        enqueue=True,
    )
    logger.add(
        stderr,
        format=f"<n>{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {identifier} {{process}} | {{file.name}}:{{function}}:{{line}} | {{message}}</n>",
        colorize=True,
        filter=lambda record: record["level"].name == "INFO",
        enqueue=True,
    )
    logger.add(
        stderr,
        format=f"<y>{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {identifier} {{process}} | {{file.name}}:{{function}}:{{line}} | {{message}}</y>",
        colorize=True,
        filter=lambda record: record["level"].name == "WARNING",
        enqueue=True,
    )
    logger.add(
        stderr,
        format=f"<r>{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {identifier} {{process}} | {{file.name}}:{{function}}:{{line}} | {{message}}</r>",
        colorize=True,
        filter=lambda record: record["level"].name == "ERROR",
        enqueue=True,
    )
    logger.add(
        stderr,
        format=f"<g>{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {identifier} {{process}} | {{file.name}}:{{function}}:{{line}} | {{message}}</g>",
        colorize=True,
        filter=lambda record: record["level"].name == "SUCCESS",
        enqueue=True,
    )
    logger.add(
        stderr,
        format=f"<b><R>{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {identifier} {{process}} | {{file.name}}:{{function}}:{{line}} | {{message}}</R></b>",
        colorize=True,
        filter=lambda record: record["level"].name == "CRITICAL",
        enqueue=True,
    )
    

    # set file logger
    if logger_config["log_to_file"]:
        if not isinstance(run_id, str):
            logger.error("Invalid logger runs_id provided to name the log files")
            logger.info("Logging to file is disabled")
        else:
            # validate directory stat
            resolved_log_path = logger_config["log_path"].joinpath(run_id)
            ensure_directory(resolved_log_path)
            resolved_log_file = resolved_log_path.joinpath(
                f"{run_id}_{identifier}.log"
            )

            if identifier == "main":
                logger.info(f"File logging to {resolved_log_path}")

            logger.add(
                resolved_log_file,
                format=f"{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {identifier} {{process}} | {{file.name}}:{{function}}:{{line}} | {{message}}",
                enqueue=True,
                filter=lambda record: record['level'].name == "TRACE",
                level="TRACE"
            )
            logger.add(
                resolved_log_file,
                format=f"{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {identifier} {{process}} | {{file.name}}:{{function}}:{{line}} | {{message}}",
                enqueue=True,
            )
