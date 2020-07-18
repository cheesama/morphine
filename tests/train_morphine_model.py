from morphine import trainer

import os, sys

trainer.train(
    file_path="nlu.md",
    batch_size=4,
    intent_optimizer_lr=5e-5,
    entity_optimizer_lr=5e-5,
    gpu_num=0,
    epochs=3
)
