# Modified from:
#   LlamaGen:  https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/sample/sample_t2i.py

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image
import argparse
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath("__file__")))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from unifork.model import *
from transformers import AutoTokenizer
import cv2


def save_image(response, path):
    os.makedirs(path, exist_ok=True)
    for i in range(response.shape[0]):
        image = response[i].permute(1, 2, 0)
        image = image.cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"image_{i}.png"), image)
        

def generate(model, batch_prompt_embs, attention_mask, 
             image_token_num_per_image=576, 
             cfg_weight = 4.0,
             temperature = 1):
    
    batch_sz = batch_prompt_embs.shape[0]
    generated_tokens = torch.zeros((batch_sz, image_token_num_per_image, 4), dtype=torch.int).cuda()
    model.model.vision_tower.vision_tower.rqtransformer.eval()
    # model.model.vision_tower.vision_tower.rqtransformer.to(torch.bfloat16)
    for i in range(image_token_num_per_image):
        outputs = model.forward_gen(inputs_embeds=batch_prompt_embs, 
                        use_cache=True, 
                        attention_mask = attention_mask, 
                        output_hidden_states = True,
                        past_key_values=outputs.past_key_values if i != 0 else None)

        hidden_state =  outputs['hidden_states'][-1][:, -1, :].unsqueeze(1)
        _, code = model.model.vision_tower.vision_tower.rqtransformer.generate(hidden_state, model.model.vision_tower.vision_tower.rqvaesiglip, cfg_weight)

        generated_tokens[:, i] = code[:,0]

        inputs_embeds = model.model.vision_tower.vision_tower.rqtransformer.embed_with_model_aux(code, model.model.vision_tower.vision_tower.rqvaesiglip)
        inputs_embeds = torch.cumsum(inputs_embeds, dim=-2)[:,:,-1,:]
        img_embeds = model.get_model().mm_projector.forward_generate(inputs_embeds)
        batch_prompt_embs = img_embeds
        attention_mask = torch.cat([attention_mask, torch.ones((batch_sz, 1), device=attention_mask.device)], dim=-1)


    image_embeds = model.model.vision_tower.vision_tower.rqtransformer.embed_with_model_aux(generated_tokens, model.model.vision_tower.vision_tower.rqvaesiglip)
    image_embeds = torch.cumsum(image_embeds, dim=-2)[:,:,-1,:]
    image_embeds = image_embeds.reshape(generated_tokens.shape[0], 24, 24, -1)
    image_embeds = image_embeds.to(torch.bfloat16)
    model.model.vision_tower.vision_tower.rqvaesiglip.to(torch.bfloat16)
    samples = model.model.vision_tower.vision_tower.rqvaesiglip.decode(image_embeds).to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255)
    samples = samples.chunk(2)[0]

    saved_folder = 'output/generated_samples'
    
    save_image(samples, saved_folder)



def main(args):
    
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load gpt model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    tokenizer.pad_token = "<|vision_pad|>"
    tokenizer.eos_token = "<|im_end|>"
    model = LlavaQwen2ForCausalLM.from_pretrained(pretrained_model_name_or_path=args.model_path, torch_dtype=torch.bfloat16).to(device)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device)
        vision_tower.to(device=device, dtype=torch.float16)
        
    model.eval()
    print(f"model is loaded")

    # Labels to condition the model with (feel free to change):
    prompts_ids, cfg_prompts_ids = [], []

    prompt_str = f"<|im_start|>USER: {args.prompt} ASSISTANT: "
    prompt_ids = tokenizer(prompt_str).input_ids
    # unconditional tokens for classifie-free guidance  
    NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
    uncond_prompt = f"<|im_start|>USER: {NEGATIVE_PROMPT}. ASSISTANT: "
    uncond_ids = tokenizer(uncond_prompt).input_ids
    prompts_ids.append(prompt_ids)
    cfg_prompts_ids.append(uncond_ids)
        
    prompts_ids.extend(cfg_prompts_ids)

    max_len = max(len(ids) for ids in prompts_ids)
    batch_input_ids = []
    for ids in prompts_ids:
        pad_len = max_len - len(ids)
        padded_ids = [tokenizer.pad_token_id] * pad_len + ids
        batch_input_ids.append(padded_ids)

    batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
    batch_prompt_embs = model.get_model().embed_tokens(batch_input_ids)

    # boi token
    boi_token = torch.full((batch_prompt_embs.size(0), 1), 0, dtype=torch.int64).to(batch_prompt_embs.device)
    boi_token_emb = model.get_model().mm_projector.token_embedding(boi_token)
    batch_prompt_embs = torch.cat((batch_prompt_embs, boi_token_emb), dim=1)
    
    attention_mask = (batch_input_ids != tokenizer.pad_token_id).long()
    boi_mask = torch.ones((attention_mask.size(0), 1), dtype=torch.long, device=attention_mask.device)  # (B, 1)
    attention_mask = torch.cat((attention_mask, boi_mask), dim=1)
    
    generate(model, batch_prompt_embs, attention_mask)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--prompt", type=str, default="an young lady wearing sheer laying on top of a round tube, in the style of photorealistic fantasies, vray tracing, money themed, tom chambers, flora borsi, cityscape.")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)