import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
import argparse
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath("__file__")))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
import json
from unifork.model import *
from transformers import AutoTokenizer
import cv2

        

def generate(model, batch_prompt_embs, attention_mask, 
             image_token_num_per_image=576, 
             cfg_weight = 3.0,
             temperature = 1):
    
    batch_sz = batch_prompt_embs.shape[0]
    generated_tokens = torch.zeros((batch_sz, image_token_num_per_image, 4), dtype=torch.int).cuda()
    model.model.vision_tower.vision_tower.rqtransformer.eval()
    # model.model.vision_tower.vision_tower.rqtransformer.to(torch.bfloat16)
    outputs = None
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
    samples = model.model.vision_tower.vision_tower.rqvaesiglip.decode(image_embeds).add_(1).mul_(127.5).clamp_(0, 255)
    samples = samples.chunk(2)[0]
    
    return samples



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
    model = LlavaQwen2ForCausalLM.from_pretrained(pretrained_model_name_or_path=args.model_path).to(device)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device)
        vision_tower.to(device=device, dtype=torch.float16)
        
    model.eval()
    print(f"model is loaded")
    
    
    # Load prompts
    metadatas = []


    with open(args.metadata_file, "r") as jsonfile:
        data = json.load(jsonfile)
        for file_id, metadata in data.items():
            metadatas.append({
                "file_name": file_id,
                "prompt": metadata["prompt"],
                "category": metadata["category"]
            })

    batch_size = 125
    global_index = 0 
    num_meta = len(metadatas)

    for batch_start in range(0, num_meta, batch_size):
        batch_metas = metadatas[batch_start: batch_start + batch_size]
        
        batch_prompts_ids = []  
        prompts_ids = []  
        cfg_prompts_ids = [] 
        for metadata in batch_metas:
            prompt = metadata['prompt']
            print(f"Prompt ({global_index: >3}/{num_meta}): '{prompt}'")
            prompt_str = f"<|im_start|>USER: {prompt}. ASSISTANT: "
            prompt_ids = tokenizer(prompt_str).input_ids
            #  classifier-free guidance
            NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
            uncond_prompt = f"<|im_start|>USER: {NEGATIVE_PROMPT}. ASSISTANT: "
            uncond_ids = tokenizer(uncond_prompt).input_ids
            prompts_ids.append(prompt_ids)
            cfg_prompts_ids.append(uncond_ids)
        
            
        batch_prompts_ids = prompts_ids + cfg_prompts_ids
        # padding
        max_len = max(len(ids) for ids in batch_prompts_ids)
        batch_input_ids = []
        for ids in batch_prompts_ids:
            pad_len = max_len - len(ids)
            padded_ids = [tokenizer.pad_token_id] * pad_len + ids
            batch_input_ids.append(padded_ids)
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
        
        batch_prompt_embs = model.get_model().embed_tokens(batch_input_ids)
        
        # add boi token
        boi_token = torch.full((batch_prompt_embs.size(0), 1), 0, dtype=torch.int64, device=batch_prompt_embs.device)
        boi_token_emb = model.get_model().mm_projector.token_embedding(boi_token)
        batch_prompt_embs = torch.cat((batch_prompt_embs, boi_token_emb), dim=1)
        
        # get attention mask
        attention_mask = (batch_input_ids != tokenizer.pad_token_id).long()
        boi_mask = torch.ones((attention_mask.size(0), 1), dtype=torch.long, device=attention_mask.device)
        attention_mask = torch.cat((attention_mask, boi_mask), dim=1)
        
        with torch.no_grad():
            samples = generate(model, batch_prompt_embs, attention_mask, cfg_weight=args.cfg_scale).to(torch.float32)
        
        os.makedirs(args.outdir, exist_ok=True)
        for i in range(batch_size):
            meta_sample = samples[i]
            image = meta_sample.permute(1, 2, 0)
            image = image.cpu().numpy().astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            metadata = batch_metas[i]
            category = metadata['category']
            file_name = metadata['file_name'] 
            
            category_dir = os.path.join(args.outdir, category)
            os.makedirs(category_dir, exist_ok=True)

            save_path = os.path.join(category_dir, f"{file_name}.jpg")
            cv2.imwrite(save_path, image)
            
            global_index += 1
        
        del samples


            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="your/model/path")
    parser.add_argument("--metadata_file",type=str, default="mjhq-30k/meta_data.json")
    parser.add_argument("--outdir", type=str, default="output/generated_samples_mjhq")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)