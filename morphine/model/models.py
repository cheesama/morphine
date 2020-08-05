from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
from torchcrf import CRF

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        intent_class_num: int,
        entity_class_num: int,
        max_seq_len: int,
        transformer_layers=2,
        num_encoder_layers=8,
        d_model=128,
        nhead=8,
        pad_token_id: int = 1,
    ):
        super(EmbeddingTransformer, self).__init__()
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        self.intent_encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead,),
            num_encoder_layers,
            LayerNorm(d_model),
        )
        self.entity_encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead,),
            num_encoder_layers,
            LayerNorm(d_model),
        )

        self.transformer_layers = transformer_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, d_model)
        self.intent_feature = nn.Linear(d_model, intent_class_num)
        self.entity_feature = nn.Linear(d_model, entity_class_num)
        self.entity_featurizer = CRF(entity_class_num, batch_first=True)

        nn.init.xavier_uniform_(self.intent_feature.weight)
        nn.init.xavier_uniform_(self.entity_feature.weight)

    def forward(self, x, entity_labels=None):
        src_key_padding_mask = x == self.pad_token_id
        embedding = self.embedding(x)
        feature = embedding + self.position_embedding(
            torch.arange(x.size(1)).type_as(x)
        ).repeat(x.size(0), 1, 1)

        intent_feature = feature
        entity_feature = feature

        for i in range(self.transformer_layers):
            # (N,S,E) -> (S,N,E) => (T,N,E) -> (N,T,E)
            intent_feature = self.intent_encoder(intent_feature.transpose(1, 0)).transpose(1, 0)
            entity_feature = self.entity_encoder(entity_feature.transpose(1, 0)).transpose(1, 0)
            #entity_feature = self.entity_encoder(entity_feature.transpose(1, 0), src_key_padding_mask=src_key_padding_mask).transpose(1, 0)
            # feature = self.encoder(feature.transpose(1, 0)).transpose(1, 0)

        intent_pred = self.intent_feature(intent_feature.mean(1))
        entity_pred = self.entity_feature(entity_feature)
        entity_crf_pred = self.entity_featurizer.decode(entity_pred)

        if entity_labels is not None:
            # CRF return log likelyhood value
            mask = src_key_padding_mask == 0
            if not mask[:, 0].all():
                entity_loss = self.entity_featurizer(
                    entity_pred, entity_labels, reduction="mean"
                )
            else:
                entity_loss = self.entity_featurizer(
                    entity_pred, entity_labels, reduction="mean", mask=mask
                )

            return intent_pred, entity_crf_pred, -entity_loss

        return intent_pred, entity_crf_pred
