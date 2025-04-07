import os
import hashlib
import pandas as pd
from tiktoken import Encoding
from torch import tensor, save, load
from torch.utils.data import Dataset
from HelloLM.utils.logger import logger
from HelloLM.utils.tools import ensure_directory


class DatasetUnit(Dataset):
    def __init__(
        self, raw_text: str, tokenizer: Encoding, max_length: int, stride: int
    ):
        super().__init__()

        # token IDs
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})

        if len(token_ids) <= max_length:
            if len(token_ids) > 1:
                self.input_ids.append(tensor(token_ids[:-1]))
                self.target_ids.append(tensor(token_ids[1:]))
        else:
            for i in range(0, len(token_ids) - max_length, stride):
                input_seq = token_ids[i : i + max_length]
                target_seq = token_ids[i + 1 : i + 1 + max_length]
                self.input_ids.append(tensor(input_seq))
                self.target_ids.append(tensor(target_seq))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


class HelloDataset(Dataset):
    @logger.catch
    def __init__(
        self,
        payload: str | list,
        column_name: str,
        tokenizer: Encoding,
        max_length: int,
        stride: int,
        use_cache: bool = True,
        cache_path: str = "dataset_cache",
    ):
        super().__init__()

        # create cache directory
        os.makedirs(cache_path, exist_ok=True)

        # dataset store
        self.datasets = []

        # parse payload input
        if isinstance(payload, str):
            payload = [payload]

        # normalize paths
        payload = [os.path.abspath(path) for path in payload]

        # cache path
        cache_key = self._generate_cache_key(payload, column_name, max_length, stride)
        cache_path = os.path.join(cache_path, f'datasets/{cache_key}.pt')

        # loop all entries to include all datasets
        if use_cache and os.path.exists(cache_path):
            logger.info(f'Loading dataset from cache: {cache_path}')
            cache_data = load(cache_path)
            self.datasets = cache_data['datasets']
            self.indices = cache_data['indices']
            self.total_samples = cache_data['total_samples']
            logger.info(f'Loaded {self.total_samples} samples from cache')
        else:
            for index, payload_entries in enumerate(payload):
                data_frame = pd.read_parquet(payload_entries)

                if column_name not in data_frame.columns:
                    logger.error("ERROR in column name resolving")
                    raise RuntimeError(
                        f"Column '{column_name}' not found in parquet {index}.\n"
                        f"Available columns: {data_frame.columns.to_list()}'"
                    )

                if len(data_frame) == 0:
                    logger.error("No enough data to load")
                    raise RuntimeError(f"Parquet {index} contains no rows")

                for raw_text in data_frame[column_name]:
                    if isinstance(raw_text, str) and len(raw_text) > 0:
                        processed = raw_text + "<|endoftext|>"
                        self.datasets.append(
                            DatasetUnit(processed, tokenizer, max_length, stride)
                        )

            # calculate valid sample number
            self.total_samples = sum(len(item) for item in self.datasets)

            logger.info(f'Total samples is: {self.total_samples}')

            if self.total_samples == 0:
                raise RuntimeError("No any valid entries in given parquet file(s)")

            # build indices
            self.indices = []
            for index, item in enumerate(self.datasets):
                for local_index in range(len(item)):
                    self.indices.append((index, local_index))
            
            # save processed dataset
            if use_cache:
                logger.info(f'Saving dataset to cache: {cache_path}')
                ensure_directory(cache_path)
                
                save({
                    "datasets": self.datasets,
                    "indices": self.indices,
                    "total_samples": self.total_samples,
                }, cache_path)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        index, local_index = self.indices[index]
        return self.datasets[index][local_index]
    
    def _generate_cache_key(self, payload, column_name, max_length, stride):
        h = hashlib.md5()
        for entry in payload:
            h.update(str(entry).encode())
            h.update(str(os.path.getmtime(entry)).encode())

        h.update(str(column_name).encode())
        h.update(str(max_length).encode())
        h.update(str(stride).encode())

        return h.hexdigest() 

