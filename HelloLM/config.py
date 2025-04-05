MODEL_CONFIG = {
    "vocab_size": 100277,
    "context_length": 2048,
    "dim_embed": 1536,
    "heads_count": 24,
    "layers_count": 24,
    "dropout_rate": 0.1,
    "qkv_bias": True,
}

TRAIN_CONFIG = {
    "learning_rate": 5e-4,
    "target_epochs": 10,
    "batch_size": 4,
    "weight_decay": 0.1,
    "dataset_cache_path": "dataset_cache"
}
