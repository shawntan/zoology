from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig



config = TrainConfig(
    data=DataConfig(
        num_train_examples=100_000,
        num_test_examples=3_000,
        vocab_size=8192,
        input_seq_len=256,
        batch_size=256,
        # cache_dir="/path/to/cache/dir"  TODO: add this
        # vocab_size=256,
        # input_seq_len=64,
        # num_train_examples=10_000,
        # num_test_examples=1_000,
        builder=FunctionConfig(
            # name="zoology.data.associative_recall.multiquery_ar",
            name="zoology.data.recent_associative_recall.multiquery_ar",
            kwargs={"num_kv_pairs": 64}
        ),
    ),
    model=ModelConfig(
        vocab_size=8192,
        max_position_embeddings=256,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            # name="zoology.mixers.sb_attention.MHA",
            n_layers=2,
            max_position_embeddings=256,
            kwargs={"dropout": 0.1, "num_heads": 1}
        )
    ),
    lr=3e-4,
    max_epochs=64,
)

configs = [config]
