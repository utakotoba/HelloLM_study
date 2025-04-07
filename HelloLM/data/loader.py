from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from HelloLM.config import ModelConfig, TrainConfig
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
    model_config: ModelConfig,
    train_config: TrainConfig,
    shuffle=True,
    drop_last=True,
    use_cache=True,
    cache_path="cache",
    rank=-1 # distributed train only
):
    # create tokenizer
    tokenizer = create_tokenizer()

    # build dataset
    dataset = HelloDataset(
        payload=payload,
        column_name=column_name,
        tokenizer=tokenizer,
        max_length=model_config['context_length'],
        stride=model_config['context_length'],
        use_cache=use_cache,
        cache_path=cache_path,
    )

    # build DistributedSampler
    if train_config['distributed']:
        dataset = DistributedSampler(
            dataset=dataset,
            num_replicas=train_config['world_size'],
            rank=rank,
            shuffle=shuffle
        )

    # create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=train_config['batch_size_per_device'],
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=train_config['dataloader_workers_num'],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader
