from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        intent_class_num: int,
        entity_class_num: int,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        pad_token_id: int = 0,
    ):
        super(EmbeddingTransformer, self).__init__()
        self.pad_token_id = pad_token_id

        self.encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation
            ),
            num_encoder_layers,
            LayerNorm(d_model),
        )

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.intent_feature = nn.Linear(d_model, intent_class_num)
        self.entity_feature = nn.Linear(d_model, entity_class_num)

    def forward(self, x):
        src_key_padding_mask = x == self.pad_token_id
        embedding = self.embedding(x)
        embedding += self.position_embedding(torch.arange(self.seq_len).repeat(x.size(0), 1).type_as(x))

        # (N,S,E) -> (S,N,E) => (T,N,E) -> (N,T,E)
        feature = self.encoder(embedding.transpose(1, 0), src_key_padding_mask=src_key_padding_mask).transpose(1,0)

        intent_pred = self.intent_feature(feature.mean(1))
        entity_pred = self.entity_feature(feature)

        return intent_pred, entity_pred


