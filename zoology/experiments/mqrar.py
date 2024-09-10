import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig, ModuleConfig

# VOCAB_SIZE = 8192
VOCAB_SIZE = 65
# input_seq_len = 768
input_seq_len = 64
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
for num_kv_pairs in [16]:
    data = DataConfig(
        num_train_examples=100_000,
        num_test_examples=3_000,
        vocab_size=VOCAB_SIZE,
        input_seq_len=input_seq_len,
        batch_size=batch_size,
        cache_dir="/workspace/shawntan/out/zoology_baselines",
        builder={
            "name": 'zoology.data.recent_associative_recall.multiquery_ar',
            # "name": "zoology.data.associative_recall.multiquery_ar",
            "kwargs": {
                "num_kv_pairs": num_kv_pairs,
                # "train_power_a": 0.01,
                # "test_power_a": 0.01,
                # "random_non_queries": False
            }
        }
    )
    # for d_model in [256]:
    for d_model in [100]:
        # for lr in  np.logspace(-4, -2, 4):
        for lr in  [3e-4]:
            for sequence_mixer in ["sb_attention", "attention"]:
                model = ModelConfig(
                    d_model=d_model,
                    n_layers=2,
                    block_type=block_type,
                    max_position_embeddings=input_seq_len,
                    vocab_size=VOCAB_SIZE,
                    sequence_mixer=MIXERS[sequence_mixer],
                    state_mixer=ModuleConfig(name='zoology.mixers.mlp.MLP', kwargs={'hidden_mult': 4}),
                )
                config = TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    weight_decay=0.1,
                    max_epochs=64,
                    run_id=f"{sequence_mixer}-seqlen{input_seq_len}-kv{num_kv_pairs}-dmodel{d_model}-lr{lr}",
                    logger=LoggerConfig(project_name="repeated_ar", entity="shawntan")
                )
                configs.append(config)
