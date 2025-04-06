from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from HelloLM.data.dataset import HelloDataset
from HelloLM.data.tokenizer import create_tokenizer


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = [seq.clone().detach() for seq in inputs]
    targets = [seq.clone().detach() for seq in targets]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets


def create_dataloader(
    payload: str | list,
    column_name: str,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    use_cache=True,
    cache_path="dataset_cache",
    return_dataset=False,
):
    # create tokenizer
    tokenizer = create_tokenizer()

    # build dataset
    dataset = HelloDataset(
        payload, column_name, tokenizer, max_length, stride, use_cache, cache_path
    )
    
    # Return just the dataset if requested (used for distributed training)
    if return_dataset:
        return dataset

    # create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader
