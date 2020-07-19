from argparse import Namespace
from collections import Counter

from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from pytorch_lightning import Trainer
from pytorch_lightning.metrics.functional import f1_score

from torchnlp.metrics import get_accuracy, get_token_accuracy
from sklearn.metrics import balanced_accuracy_score

from .dataset.intent_entity_dataset import (
    RasaIntentEntityDataset,
    token_concat_collate_fn,
)
from .model.models import EmbeddingTransformer

import os, sys
import multiprocessing

import torch
import torch.nn as nn
import pytorch_lightning as pl


class MorphineClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        if type(self.hparams) == dict:
            self.hparams = Namespace(**self.hparams)

        self.dataset = RasaIntentEntityDataset(markdown_lines=self.hparams.nlu_data)

        self.model = EmbeddingTransformer(
            len(self.dataset.vocab_dict),
            len(self.dataset.intent_dict),
            len(self.dataset.entity_dict),
        )
        self.train_ratio = self.hparams.train_ratio
        self.batch_size = self.hparams.batch_size
        self.optimizer = self.hparams.optimizer
        self.intent_optimizer_lr = self.hparams.intent_optimizer_lr
        self.entity_optimizer_lr = self.hparams.entity_optimizer_lr

        self.intent_loss_fn = nn.CrossEntropyLoss()

        # ignore O tag class label to figure out entity imbalance distribution
        #self.entity_loss_fn = nn.CrossEntropyLoss(ignore_index=self.dataset.pad_token_id)
        self.entity_loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([0.1] + [1.0] * (len(self.dataset.get_entity_idx()) - 1)))

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        train_length = int(len(self.dataset) * self.train_ratio)

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_length, len(self.dataset) - train_length],
        )

        '''
        intent_sampling_weights = [
            1 / item[1]
            for item in sorted(
                Counter(
                    [each_dataset[1] for each_dataset in self.train_dataset]
                ).items()
            )
        ]
        sampling_weights = [
            intent_sampling_weights[item[1]] for item in self.train_dataset
        ]
        self.sampler = WeightedRandomSampler(
            sampling_weights, len(sampling_weights), replacement=False
        )
        '''

        self.hparams.intent_label = self.get_intent_label()
        self.hparams.entity_label = self.get_entity_label()

    def get_intent_label(self):
        self.intent_dict = {}
        for k, v in self.dataset.intent_dict.items():
            self.intent_dict[str(v)] = k
        return self.intent_dict

    def get_entity_label(self):
        self.entity_dict = {}
        for k, v in self.dataset.entity_dict.items():
            self.entity_dict[str(v)] = k
        return self.entity_dict

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count() * 2,
            collate_fn=token_concat_collate_fn,
            #sampler=self.sampler,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count() * 2,
            collate_fn=token_concat_collate_fn,
        )
        return val_loader

    def configure_optimizers(self):
        optimizers = [
            eval(f"{self.optimizer}(self.parameters(), lr={self.intent_optimizer_lr})"),
            eval(f"{self.optimizer}(self.parameters(), lr={self.entity_optimizer_lr})"),
            # eval(f"{self.optimizer}(self.intent_center_loss_fn.parameters(), lr={self.intent_optimizer_lr})")
        ]

        schedulers = [
            {
                "scheduler": ReduceLROnPlateau(optimizers[0], patience=1, factor=0.5),
                "monitor": "val_intent_f1",
                "interval": "epoch",
                "frequency": 1,
            },
            {
                "scheduler": ReduceLROnPlateau(optimizers[1], patience=1, factor=0.5),
                "monitor": "val_entity_acc",
                "interval": "epoch",
                "frequency": 1,
            },
        ]

        return optimizers, schedulers

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.model.train()

        tokens, intent_idx, entity_idx = batch
        intent_pred, entity_pred = self.forward(tokens)

        if torch.isnan(tokens).sum().item() > 0:
            print ('tokens error')
        if torch.isnan(intent_idx).sum().item() > 0:
            print ('intent_idx error')
        if torch.isnan(entity_idx).sum().item() > 0:
            print ('entity_idx error')
        if torch.isnan(intent_pred).sum().item() > 0:
            print ('intent_pred error')
        if torch.isnan(entity_pred).sum().item() > 0:
            print ('entity_pred error')

        intent_acc = get_accuracy(intent_pred.argmax(1), intent_idx)[0]
        intent_f1 = f1_score(intent_pred.argmax(1), intent_idx)

        entity_acc = balanced_accuracy_score(
            entity_idx.cpu().tolist()[0], entity_pred.max(2)[1].cpu().tolist()[0]
        )

        tensorboard_logs = {
            "train/intent/acc": intent_acc,
            "train/intent/f1": intent_f1,
            "train/entity/acc": entity_acc,
        }

        if optimizer_idx == 0:
            intent_loss = self.intent_loss_fn(intent_pred, intent_idx.long())
            tensorboard_logs["train/intent/loss"] = intent_loss

            return {
                "loss": intent_loss,
                "log": tensorboard_logs,
            }

        if optimizer_idx == 1:
            entity_loss = self.entity_loss_fn(entity_pred.transpose(1, 2), entity_idx.long())
            tensorboard_logs["train/entity/loss"] = entity_loss

            return {
                "loss": entity_loss,
                "log": tensorboard_logs,
            }

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        tokens, intent_idx, entity_idx = batch
        intent_pred, entity_pred = self.forward(tokens)

        intent_acc = get_accuracy(intent_pred.argmax(1), intent_idx)[0]
        intent_f1 = f1_score(intent_pred.argmax(1), intent_idx)

        entity_acc = balanced_accuracy_score(
            entity_idx.cpu().tolist()[0], entity_pred.max(2)[1].cpu().tolist()[0]
        )

        intent_loss = self.intent_loss_fn(intent_pred, intent_idx.long(),)
        entity_loss = self.entity_loss_fn(
            entity_pred.transpose(1, 2), entity_idx.long(),
        )

        return {
            "val_intent_acc": torch.Tensor([intent_acc]),
            "val_intent_f1": torch.Tensor([intent_f1]),
            "val_entity_acc": torch.Tensor([entity_acc]),
            "val_loss": intent_loss + entity_loss,  # + intent_center_loss,
        }

    def validation_epoch_end(self, outputs):
        avg_intent_acc = torch.stack([x["val_intent_acc"] for x in outputs]).mean()
        avg_intent_f1 = torch.stack([x["val_intent_f1"] for x in outputs]).mean()
        avg_entity_acc = torch.stack([x["val_entity_acc"] for x in outputs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        print(f"epoch : {self.current_epoch}")
        print(f"intent_acc : {avg_intent_acc}, intent_f1 : {avg_intent_f1}")
        print(f"entity_acc : {avg_entity_acc}, val_loss : {avg_loss}")
        print()

        tensorboard_logs = {
            "val/intent_acc": avg_intent_acc,
            "val/intent_f1": avg_intent_f1,
            "val/entity_acc": avg_entity_acc,
            "val/val_loss": avg_loss,
        }

        return {
            "val_loss": avg_loss,
            "val_intent_acc": avg_intent_acc,
            "val_intent_f1": avg_intent_f1,
            "val_entity_acc": avg_entity_acc,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
