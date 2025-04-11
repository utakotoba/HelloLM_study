import os
from pathlib import Path
from datetime import datetime
from loguru import logger


@logger.catch
def ensure_directory(target):
    if not isinstance(target, Path):
        target = Path(target)
    resolved = target.resolve()
    if not resolved.exists():
        os.makedirs(resolved, exist_ok=True)
    try:
        if not resolved.is_dir():
            raise NotADirectoryError(f"{target} is not a valid directory")
    except NotADirectoryError as e:
        logger.exception(e)
    return resolved

def generate_run_id():
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return timestamp
