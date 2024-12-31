import torch
import torch.nn as nn
import torch.nn.functional as F


class HomographyRegression(nn.Module):
    def __init__(self, output_dim=16):
        super().__init__()
        # self.pre_norm = LayerNorm(576)
        encoder_layer = nn.TransformerEncoderLayer(d_model=576, nhead=12, dim_feedforward=2048, activation="gelu", dropout=0.)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.pe = nn.Parameter(torch.zeros(1, 576, 576),requires_grad=True)
        output_dim_last_conv = 576*576
        self.linear = nn.Linear(output_dim_last_conv, output_dim)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x += self.pe
        x = x.permute(2,0,1)
        x = self.encoder(x)
        x1 = x.clone()
        x = self.encoder2(x)
        x = x + x1
        x = x.permute(1,2,0)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        return x.reshape(B, 8, 2)
