from morphine import trainer

import os, sys

trainer.train(
    file_path="nlu.md",
    batch_size=256,
    intent_optimizer_lr=1e-4,
    entity_optimizer_lr=1e-4,
    gpu_num=0,
    epochs=15
)
