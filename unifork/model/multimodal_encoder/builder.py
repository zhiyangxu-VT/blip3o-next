import os
from .clip_encoder import CLIPVisionTower
from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler, SanaTransformer2DModel
import torch

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("google"):
        if not use_s2:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_sana(vision_tower_cfg, **kwargs):
    sana = SanaTransformer2DModel.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_512px_diffusers", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    return sana

def build_vae(vision_tower_cfg, **kwargs):
    vae = AutoencoderDC.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_512px_diffusers", subfolder="vae", torch_dtype=torch.bfloat16
    )
    return vae