import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig

VOCAB_SIZE = 8_192
input_seq_len = 512
batch_size = 256
block_type = "TransformerBlock"

MIXERS = {
    "attention": dict(
        name="zoology.mixers.attention.MHA",
        kwargs={
            "dropout": 0.1,
            "num_heads": 1
        },
    ),
    "sb_attention": dict(
        name="zoology.mixers.sb_attention.MHA",
        kwargs={
            "dropout": 0.1,
            "num_heads": 1
        },
    ),
}


configs = []
for num_kv_pairs in [8, 16, 64, 128]:
    data = DataConfig(
        num_train_examples=100_000,
        num_test_examples=3_000,
        vocab_size=VOCAB_SIZE,
        input_seq_len=input_seq_len,
        batch_size=batch_size,
        cache_dir="/workspace/shawntan/out/zoology_baselines",
        builder={
            "name": "zoology.data.associative_recall.multiquery_ar",
            "kwargs": {
                "num_kv_pairs": num_kv_pairs,
                "train_power_a": 0.01,
                "test_power_a": 0.01,
                "random_non_queries": False
            }
        }
    )
    for d_model in [256, 512]:
        for lr in  np.logspace(-4, -2, 4):
            for sequence_mixer in ["sb_attention", "attention"]:
                model = ModelConfig(
                    d_model=d_model,
                    n_layers=2,
                    block_type=block_type,
                    max_position_embeddings=input_seq_len if sequence_mixer == "attention" else 0,
                    vocab_size=VOCAB_SIZE,
                    sequence_mixer=MIXERS[sequence_mixer],
                    state_mixer=dict(name="torch.nn.Identity", kwargs={})
                )
                config = TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    max_epochs=64,
                    run_id=f"{sequence_mixer}-seqlen{input_seq_len}-kv{num_kv_pairs}-dmodel{d_model}-lr{lr}",
                    logger=LoggerConfig(project_name="repeated_ar", entity="shawntan")
                )
                configs.append(config)
