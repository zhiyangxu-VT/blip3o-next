from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch.nn.functional as F
import torch
import transformers
import random
from torch.utils.data import Dataset
import torch
import random
from datasets import load_dataset
from torchvision.transforms import v2

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
project_root1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
project_root2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root1)
sys.path.insert(0, project_root2)

from configs.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import ConcatDataset
from typing import Any, Tuple, Callable, Optional
from unifork.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from unifork.train.llava_trainer import LLaVATrainer
from unifork import conversation as conversation_lib
from unifork.model import *
from transformers import logging
# from data_utils import DatasetCeph , DatasetIntern


local_rank = None
os.environ['RANK'] = os.environ['SLURM_PROCID']
os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
os.environ['MASTER_PORT'] = os.environ['MASTER_PORT']
os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vq_vae_path: Optional[str] = field(default='')
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="same")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    imagenet_root: Optional[str] = field(default='')
    data_meta_path: Optional[str] = field(default='data_internvl_pretrain.json')
    label_mapping_path: Optional[str] = field(default='configs/imagenet_label_mapping')
    image_size: int = 384
    training_stage: float=0.0


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    logging_steps: int = 10
    report_to = "none"
    model_max_length: int = field(
        default=1350,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return



def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# class ImageNetGen(ImageFolder):
#     def __init__(
#         self, 
#         root: str, 
#         version: str, 
#         tokenizer: Any, 
#         image_processor: Any, 
#         transform: Optional[Callable] = None, 
#         target_transform: Optional[Callable] = None, 
#         loader: Callable[[str], Any] = default_loader,
#         is_valid_file: Optional[Callable[[str], bool]] = None, 
#     ):
#         super().__init__(
#             root,
#             transform=transform,
#             target_transform=target_transform,
#             loader=loader,
#             is_valid_file=is_valid_file,
#         )

#         self.preprocess_version = version
#         self.tokenizer = tokenizer
#         self.image_processor = image_processor
        
    
#     def __getitem__(self, index: int) -> Tuple[Any, str]:
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (sample, labels_text) where labels_text is the formatted text label.
#         """
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
        
#         image = self.image_processor.preprocess(sample, return_tensors='pt', gen_task=True)['pixel_values'][0]
#         # Get class name and formatted text
#         class_name = IMAGENET_CLASSNAMES[target]
#         template = random.choice(OPENAI_IMAGENET_TEMPLATES)
#         labels_text = template(class_name)
        
#         labels_text = "<|im_start|>{}".format(labels_text)
#         prompt_id = self.tokenizer(labels_text).input_ids
#         input_ids = torch.tensor(prompt_id + [IMAGE_TOKEN_INDEX])
#         labels = torch.tensor([IGNORE_INDEX]*len(prompt_id) + [IMAGE_TOKEN_INDEX])

#         return dict(input_ids=input_ids, labels=labels, image=image, is_gen=True)


class ImageNetGen(Dataset):
    def __init__(
        self,
        hf_data_path: str,
        tokenizer: Any,
        image_processor: Any,
        version: str = "default",
        split: str = "train",
        **load_dataset_kwargs
    ):
        """
        Args:
            hf_data_path (str): HuggingFace dataset repo path (e.g., 'imagenet-1k' or custom path).
            tokenizer (Any): Tokenizer instance.
            image_processor (Any): Image processor (e.g., CLIP processor).
            version (str): Preprocessing version/tag.
            split (str): Split to load, e.g. "train" or "validation".
            **load_dataset_kwargs: Any additional arguments for `load_dataset`.
        """
        super().__init__()
        self.dataset = load_dataset(
            hf_data_path,
            split=split,
            **load_dataset_kwargs
        )
        self.preprocess_version = version
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.target_transform = v2.Compose(
        [
            v2.Resize(512),
            v2.CenterCrop(512),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # Most HF ImageNet datasets have fields like 'image' (PIL.Image) and 'label' (int)
        sample = self.dataset[idx]
        image = sample['image'].convert("RGB")
        target = sample['label']
        # Preprocess image
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt', gen_task=True)['pixel_values'][0]
        target_image_tensor = self.target_transform(image)
        # Prepare prompt and labels as before
        class_name = IMAGENET_CLASSNAMES[target]
        template = random.choice(OPENAI_IMAGENET_TEMPLATES)
        labels_text = "<|im_start|>{}".format(template(class_name))
        prompt_id = self.tokenizer(labels_text).input_ids
        input_ids = torch.tensor(prompt_id + [IMAGE_TOKEN_INDEX])
        labels = torch.tensor([IGNORE_INDEX]*len(prompt_id) + [IMAGE_TOKEN_INDEX])

        return dict(input_ids=input_ids, labels=labels, image=image_tensor, target_image=target_image_tensor, is_gen=True)


class ImageNetUnd(ImageFolder):
    def __init__(
        self, 
        root: str, 
        version: str, 
        tokenizer: Any, 
        image_processor: Any, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None, 
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None, 
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.preprocess_version = version
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    
    def __getitem__(self, index: int) -> Tuple[Any, str]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, labels_text) where labels_text is the formatted text label.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        image = self.image_processor.preprocess(sample, return_tensors='pt')['pixel_values'][0]
        # Get class name and formatted text
        class_name = IMAGENET_CLASSNAMES[target]
        template = random.choice(OPENAI_IMAGENET_TEMPLATES)
        labels_text = template(class_name)
        
        prompt_id = self.tokenizer(labels_text).input_ids
        input_ids = torch.tensor([151644] + [IMAGE_TOKEN_INDEX] + prompt_id)
        labels = torch.tensor([IGNORE_INDEX]*2 + prompt_id)

        return dict(input_ids=input_ids, labels=labels, image=image, is_gen=False)


@dataclass
class DataCollatorForCombinedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        images_gen = [item['image'] for item in instances if item['is_gen']]
        images_und = [item['image'] for item in instances if not item['is_gen']]
        tar_images_gen = [item['target_image'] for item in instances if item['is_gen']]
        tar_images_gen = torch.stack(tar_images_gen, dim=0) if tar_images_gen else None
        images_gen = torch.stack(images_gen, dim=0) if images_gen else None
        images_und = torch.stack(images_und, dim=0) if images_und else None
        is_gen = torch.tensor([item['is_gen'] for item in instances], dtype=torch.bool)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            is_gen = is_gen,
            images_gen = images_gen,
            tar_images_gen = tar_images_gen,
            images_und = images_und
        )
        
        return batch


def build_datasets(
    data_args,
    tokenizer,
    version,
):

    with open(data_args.data_meta_path, "r", encoding="utf-8") as f:
        ds_config = json.load(f)

    all_datasets = []

    #########  build generation datasets #########
    if data_args.training_stage == 1.5:
        dataset_gen = DatasetCeph(tokenizer, data_args.image_processor, dataset_name='laion')
        all_datasets.append(dataset_gen)
    elif data_args.training_stage == 2.0:
        dataset_names = ['your/dataset/name/list']
        for dataset_name in dataset_names:
            print(f"[combine_gen_datasets] Building dataset: {dataset_name}")
            dataset_gen = DatasetCeph(tokenizer, data_args.image_processor, dataset_name=dataset_name)
            all_datasets.append(dataset_gen)
    elif data_args.training_stage == 3.0 or data_args.training_stage == 4.0:  
        dataset_names = ['your/dataset/name/list']
        for dataset_name in dataset_names:
            print(f"[combine_gen_datasets] Building dataset: {dataset_name}")
            dataset_gen = DatasetCeph(tokenizer, data_args.image_processor, dataset_name=dataset_name)
            all_datasets.append(dataset_gen)

    #########  build understanding datasets  #########
    if data_args.training_stage == 1.5:
        dataset_und = DatasetCeph(tokenizer, data_args, dataset_name='und_coyo')
        all_datasets.append(dataset_und)
    elif data_args.training_stage == 2.0 or data_args.training_stage == 3.0 or data_args.training_stage == 5.0:
        for ds_name, ds_info in ds_config.items():
            annotation_path = ds_info["annotation"] 
            root_path = ds_info.get("root", "")   
            repeat_time = ds_info.get("repeat_time", 1)
            print(f"[combine_und_datasets] Building dataset: {ds_name}")
            dataset = DatasetIntern(
                data_path=annotation_path,    
                tokenizer=tokenizer,
                data_args=data_args,
                root_path=root_path,
                repeat_time=repeat_time       
            )
            all_datasets.append(dataset)
    
    combined_dataset = ConcatDataset(all_datasets)
    total_len = len(combined_dataset)
    print(f" [combine_datasets] Total combined length: {total_len}")

    return combined_dataset

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                version,
                                per_device_train_batch_size,) -> Dict:
    
    if data_args.training_stage == 1.0:
        # all_datasets = []
        # train_dataset_und = ImageNetUnd(data_args.imagenet_root, version, tokenizer, data_args.image_processor)
        # all_datasets.append(train_dataset_und)
        train_dataset_gen = ImageNetGen(data_args.imagenet_root, tokenizer, data_args.image_processor, version=version, split='train')
        # all_datasets.append(train_dataset_gen)
        # train_dataset = ConcatDataset(all_datasets)
        train_dataset = train_dataset_gen
    else:
        train_dataset = build_datasets(data_args=data_args,
                                       tokenizer=tokenizer,
                                       version=version)
    
    data_collator = DataCollatorForCombinedDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    os.environ["WANDB_DISABLED"] = "true"
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}

    if model_args.vision_tower is not None:
        if model_args.tune_mm_mlp_adapter:
            model = LlavaQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            # use layers_parameters to initialize the gen_fork_layers
            start_idx = len(model.model.layers) // 2
            for idx, layer in enumerate(model.model.layers[start_idx:]):
                layer_dict = layer.state_dict()
                model.model.gen_fork_layers[idx].load_state_dict(layer_dict)
            norm_dict = model.model.norm.state_dict()
            model.model.norm_gen.load_state_dict(norm_dict)
            print('init gen_fork_layers from model.layers: done!!!')
        else:
            model, loading_info = LlavaQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                output_loading_info=True,
                **bnb_model_from_pretrained_args
            )
            if loading_info:
                missing_keys = loading_info.get("missing_keys", [])
                unexpected_keys = loading_info.get("unexpected_keys", [])

                # Print missing keys and their contents
                print("=== Missing Keys ===")
                for key in missing_keys:
                    print(f"Key Name: {key}")
                    print(f"Content: {model.state_dict().get(key, 'Not Found')}")

                # Print unexpected keys and their contents
                print("=== Unexpected Keys ===")
                for key in unexpected_keys:
                    print(f"Key Name: {key}")
                    print(f"Content: {model.state_dict().get(key, 'Not Found')}")
    else:
        model = transformers.Qwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    tokenizer.pad_token = "<|vision_pad|>" #tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=model.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.model.vision_tower.vision_tower.eval()
        
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter or data_args.training_stage == 1.5:
            model.requires_grad_(False)
            model.model.vision_tower.vision_tower.rqtransformer.requires_grad_(True)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        elif data_args.training_stage == 2.0 or data_args.training_stage == 3.0:
            model.requires_grad_(True)
            model.model.vision_tower.vision_tower.requires_grad_(False)
            model.model.vision_tower.vision_tower.rqtransformer.requires_grad_(True)
        elif data_args.training_stage == 4.0:
            model.requires_grad_(False)
            model.model.vision_tower.vision_tower.rqtransformer.requires_grad_(True)
            for p in model.model.gen_fork_layers.parameters():
                p.requires_grad = True
            for p in model.model.norm_gen.parameters():
                p.requires_grad = True
        elif data_args.training_stage == 5.0:
            model.requires_grad_(False)
            for p in model.model.layers[12:].parameters():
                p.requires_grad = True
            for p in model.model.norm.parameters():
                p.requires_grad = True
            
            
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              version=model_args.version,
                                              per_device_train_batch_size = training_args.per_device_train_batch_size,)
    

    print("Training parameters:")
    print([name for name, param in model.named_parameters() if param.requires_grad])
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)  # data_module   data_module_gen

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
