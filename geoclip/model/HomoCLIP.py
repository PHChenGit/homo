import torch
import torch.nn as nn
import torch.nn.functional as F
from geoclip.model.GeoCLIP import GeoCLIP

def compute_similarity(features_a, features_b):
    b, c, h, w = features_a.shape
    print(f"features_a shape: {features_a.shape}")
    print(f"features_b shape: {features_b.shape}")
    features_a = features_a.transpose(2, 3).contiguous().view(b, c, h*w)
    print(f"transpose features_a shape: {features_a.shape}")
    features_b = features_b.view(b, c, h*w).transpose(1, 2)
    print(f"transpose features_b shape: {features_b.shape}")
    features_mul = torch.bmm(features_b, features_a)
    print(f"mul: {features_mul.shape}")
    print('{:.3f}MB'.format(torch.cuda.memory_allocated()/1024**3))
    correlation_tensor = features_mul.view(b, h, w, h*w).transpose(2, 3).transpose(1, 2)
    correlation_tensor = F.normalize(F.relu(correlation_tensor), p=2, dim=1)
    return correlation_tensor


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
    def forward(self, query_img, ref_img):
        B, C, H, W = query_img.shape
        x = query_img.view(B, C, -1)
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


class HomoCLIP(nn.Module):
    def __init__(self):
        super(HomoCLIP, self).__init__()
        self.homo_regression = HomographyRegression()

    def forward(self, query_imgs, ref_imgs, ):
        x = compute_similarity(query_imgs, ref_imgs)
        rotation_matrix = self.homo_regression(x)
        return rotation_matrix

