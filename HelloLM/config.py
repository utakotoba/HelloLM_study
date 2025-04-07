from typing import TypedDict


class ModelConfig(TypedDict):
    vocab_size: int
    context_length: int
    dim_embed: int
    heads_count: int
    layers_count: int
    dropout_rate: float
    qkv_bias: bool


MODEL_CONFIG: ModelConfig = {
    "vocab_size": 100277,
    "context_length": 2048,
    "dim_embed": 1536,
    "heads_count": 24,
    "layers_count": 24,
    "dropout_rate": 0.1,
    "qkv_bias": True,
}


class TrainConfig(TypedDict):
    learning_rate: float
    batch_size_per_device: int
    target_epochs: int
    weight_decay: float
    warmup_steps: int
    gradient_accumulation_steps: int
    use_mixed_precision: bool
    memory_efficient: bool

    # evaluation
    evaluation_step: int
    evaluation_iter: int
    test_output_context: str

    # cache and backup
    cache_path: str
    backup_path: str
    backup_steps: int
    max_backup_nums: int

    # dataset
    train_datasets_path: str | list
    train_datasets_column_name: str
    validation_datasets_path: str | list
    validation_datasets_column_name: str
    dataloader_workers_num: int

    # distributed
    distributed: bool
    world_size: int

    # reproduction
    seed: int


TRAIN_CONFIG: TrainConfig = {
    "learning_rate": 5e-4,
    "target_epochs": 2,
    "batch_size_per_device": 4,
    "weight_decay": 0.1,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 1,
    "use_mixed_precision": True,
    "memory_efficient": True,
    "cache_path": "cache",
    "backup_path": "ckpts",
    "max_backup_nums": 5,
    "backup_steps": 400,
    "train_datasets_path": [
        "data/wikitext/train-00000-of-00002.parquet",
        "data/wikitext/train-00001-of-00002.parquet",
    ],
    "train_datasets_column_name": "text",
    "validation_datasets_path": "data/wikitext/validation-00000-of-00002.parquet",
    "validation_datasets_column_name": "text",
    "dataloader_workers_num": 2,
    "distributed": False,
    "world_size": 1,
    "seed": 76,
}
