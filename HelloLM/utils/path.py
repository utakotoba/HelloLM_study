import os
from HelloLM.utils.logger import logger

@logger.catch
def ensure_directory(path: str):
    resolved = path
    if not os.path.isabs(path):
        resolved = os.path.join(os.getcwd(), path)
    
    if not os.path.exists(path):
        os.makedirs(resolved, exist_ok=True)
