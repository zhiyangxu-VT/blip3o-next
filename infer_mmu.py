import argparse
import torch
import os

from unifork.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from unifork.conversation import conv_templates, SeparatorStyle
from unifork.model.builder import load_pretrained_model
from unifork.utils import disable_torch_init
from unifork.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image




def eval_model(args):

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.pad_token = "<|vision_pad|>"
    tokenizer.eos_token = "<|im_end|>"
    
    
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + args.query
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + args.query

    image = Image.open(args.image_path).convert('RGB')
    
    conv = None
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image_tensor = process_images([image], image_processor, model.config)[0].to(torch.bfloat16)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).cuda(),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=256,
            use_cache=True,
            )
    
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    print(outputs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--query", type=str, default="Do you think the image is unusual or not? Why?")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
