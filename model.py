import torch
import torch.nn as nn
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Custom Transformer Encoder Layer with Rotary Position Embeddings
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def with_pos_embed(self, tensor, freqs):
        if freqs is None:
            return tensor
        seq_len, batch_size, feature_dim = tensor.shape

        # Adjust freqs to match the shape [seq_len, feature_dim]
        freqs = freqs[:seq_len, :]

        # Ensure freqs has the same feature_dim as tensor
        if freqs.shape[1] != feature_dim:
            raise ValueError(f"Frequency dimension mismatch: {freqs.shape[1]} vs {feature_dim}")

        # Expand freqs to match the batch size
        freqs = freqs.unsqueeze(1).expand(-1, batch_size, -1)
        return tensor + freqs


    def forward(self, src, src_mask=None, src_key_padding_mask=None, freqs=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, freqs)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.ninp = ninp 
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, ntoken)

        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(ninp, nhead, nhid, dropout)
            for _ in range(nlayers)
        ])

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, freqs=None):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        for layer in self.layers:
            src = layer(src, freqs=freqs)

        output = self.decoder(src)
        return output
