import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)



class VisionProjector(nn.Module):
    def __init__(self, config):
        super(VisionProjector, self).__init__()

        self.config = config
        self.linear1 = nn.Linear(1024, self.config.hidden_size)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.token_embedding = nn.Embedding(2, self.config.hidden_size)
        

    def forward(self, x):
        boi_emb, eoi_emb = self.token_embedding(torch.tensor([0, 1]).to(x.device))
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = torch.cat([boi_emb.unsqueeze(0).expand(x.size(0), -1, -1), x, eoi_emb.unsqueeze(0).expand(x.size(0), -1, -1)], dim=1)
        return x
    
    def forward_generate(self, x):
        
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        
        return x
    
def build_vision_projector(config, delay_load=False, **kwargs):
    
    return VisionProjector(config)

