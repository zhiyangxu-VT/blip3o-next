import torch
import torch.nn as nn
from .image_processing_vlm import VLMImageProcessor
from .rqvaesigliptransformer.modeling_rqvaesigliptransformer import RQVAESIGLIPTransformer
import os



class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'same')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        # self.load_model()
        # else:
        #     self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('vision_tower is already loaded, `load_model` called again, skipping.')
            return

        self.image_processor = VLMImageProcessor.from_pretrained('configs/preprocessor_config.json')
        self.vision_tower = RQVAESIGLIPTransformer.from_pretrained(self.vision_tower_name,local_files_only=True, ignore_mismatched_sizes=True)
        self.vision_tower.rqtransformer.apply(
            lambda module: module.reset_parameters() if hasattr(module, "reset_parameters") else None)

        # for stage 2/3
        # self.vision_tower.eval()
        # self.vision_tower.requires_grad_(False)
        # for param in self.vision_tower.rqtransformer.parameters():
        #     param.requires_grad = True
        
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        
        output = self.vision_tower.rqvaesiglip.encode_image(images)
        image_features, tokens = output[-1], output[-2]

        bs, patch_size, _, dim = image_features.shape
        image_features = torch.reshape(image_features, [bs, patch_size**2, dim])
        tokens = torch.reshape(tokens, [bs, patch_size**2, tokens.shape[-1]])
        return image_features, tokens

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vq_vae.dtype

    @property
    def device(self):
        return self.vq_vae.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vq_vae.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
