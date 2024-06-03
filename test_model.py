import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SpectralPatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, embedding_dim: int, device: str) -> None:
        super(SpectralPatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embedding = nn.Linear(patch_size, embedding_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f'SpectralPatchEmbedding input shape: {x.shape}')
        x = x.unfold(1, self.patch_size, self.patch_size).contiguous()
        print(f'After unfold: {x.shape}')
        x = x.view(x.size(0), -1, self.patch_size)
        print(f'After view: {x.shape}')
        x = self.embedding(x)
        print(f'After embedding: {x.shape}')
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, device: str, max_len: int = 5000, base: float = 10000.0) -> None:
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.base = base
        
        pe = torch.zeros(max_len, d_model,device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f'in positionencoding {x.shape}')
        x = x + self.pe[:x.size(1)]
        print(f'after positionencoding {x[0,0].shape}')
        return x









dummy_input = torch.randn(1,177).to(device)


class models(nn.Module):
    def __init__(self, patch_size: int, embedding_dim: int, device: str,num_heads: int, dim_feedforward: int,
                 num_encoder_layers: int, activation_fn=F.gelu) -> None:
        super(models,self).__init__()
        self.spectralEmbedding= SpectralPatchEmbedding(patch_size = patch_size, embedding_dim= embedding_dim, device=device)
        self.positionalEncoding= PositionalEncoding(embedding_dim, device=device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
                                                    activation=activation_fn, device=device,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x: torch.tensor):
        x= self.spectralEmbedding(x)
        x= self.positionalEncoding(x)
        # x= x.permute(1,0,2)
        x= self.transformer_encoder(x)
        print(f'After Transformer: {x.shape}')
        return x
model= models(patch_size=8, embedding_dim= 64, device=device,num_heads=4, dim_feedforward=4,num_encoder_layers=4)
# x=nn.Linear(8,64,device="cuda")
# print(x(dummy_input))
print(model(dummy_input))
summary(model, input_size=(1,177))


# x=nn.Linear(8,64,device="cuda")
# print(x(dummy_input))
