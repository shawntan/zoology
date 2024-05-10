from pathlib import Path

import wandb
from torch.nn import Module

from zoology.config import LoggerConfig, TrainConfig

class WandbLogger:
    def __init__(self, config: TrainConfig):
        if config.logger.project_name is None or config.logger.entity is None:
            print("No logger specified, skipping...")
            self.no_logger = True
            return
        self.no_logger = False
        print("WandB logger")
        self.run = wandb.init(
            name=config.run_id,
            entity=config.logger.entity,
            project=config.logger.project_name, 
        )
        wandb.run.log_code(
            root=str(Path(__file__).parent.parent),
            include_fn=lambda path, root: path.endswith(".py")
        )

    def log_config(self, config: TrainConfig):
        if self.no_logger:
            return
        self.run.config.update(config.model_dump(), allow_val_change=True)

    def log_model(self, model: Module):
        if self.no_logger:
            return
        wandb.watch(model)

    def log(self, metrics: dict):
        if self.no_logger:
            return
        wandb.log(metrics)

    def finish(self):
        if self.no_logger:
            return
        self.run.finish()


