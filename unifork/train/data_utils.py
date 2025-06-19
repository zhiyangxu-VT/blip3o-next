import json
import re
import os
import copy
import numpy as np
import torch
import hashlib
import transformers
import tokenizers
import random
from torch.utils.data import Dataset
from io import BytesIO
from petrel_client.client import Client
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError
from typing import Any, Dict, Sequence
from unifork.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from unifork.mm_utils import tokenizer_image_token
from unifork import conversation as conversation_lib


conversation_lib.default_conversation = conversation_lib.conv_templates['v1']


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

    
    
def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


# Reuse the existing helper functions:
def process_anno_laion_coyo(s):
    s = s.replace("langchao_new:", "")
    s = s.replace("langchao:", "")
    return s


def get_prompt_laion_coyo(conversations):
    for convo in conversations:
        if convo.get("from") == "gpt":
            return convo.get("value")
    return None


def load_annotation_lines(client, anno_file):
    if client:
        file_content = client.get(anno_file)
        file_content = file_content.decode("utf-8")
        lines = file_content.splitlines()
    else:
        with open(anno_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
    data_list = []
    for line in lines:
        try:
            data = json.loads(line)
            data_list.append(data)
        except json.JSONDecodeError as e:
            print(f"loading error: {e}, on line: {line}")

    return data_list


def read_jsonl_from_s3(s3_path, client):

    try:
        file_content = client.get(s3_path, no_cache=True)
        file_content = file_content.decode("utf-8")

        data_list = []
        for i, line in enumerate(file_content.splitlines()):
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                # print(f"⚠️ [WARNING] Skipping corrupted JSON line {i+1}: {e}")
                continue
        return data_list
    
    except Exception as e:
        print(f"❌ Failed to read {s3_path}: {e}")
        return None



def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()  # in qwen2, pad_token = eos_token
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation



def preprocess_multimodal(
    sources: Sequence[str],
    data_args: Any
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources



def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(conversations)
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        target[:1] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) -1
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids) -1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    s2: bool = False,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    source = sources[0]
    if not s2:
        assert len(source) == 2
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
    conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
    conversations.append(conversation)
    # tokenize conversations
    # print(conversations)
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)



def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    s2 = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer, s2)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


        
class DatasetIntern(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, 
        data_path: str, 
        tokenizer: transformers.PreTrainedTokenizer, 
        data_args: Any,
        root_path: str = "",
        repeat_time=1.0,
    ):
        super(DatasetIntern, self).__init__()
        
        self.client = Client("path/to/petreloss.conf")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.root_path = root_path
        
        self.list_data_dict = read_jsonl_from_s3(data_path, self.client)
        original_len = len(self.list_data_dict)
        
        print(f" Original length: {original_len}")
        
        if repeat_time < 1:
            # If repeat_time is less than 1, select a portion of the data
            self.list_data_dict = self.list_data_dict[:int(len(self.list_data_dict) * repeat_time)]
        if repeat_time > 1:
            assert isinstance(repeat_time, int)
            # Repeat the list if repeat_time is greater than 1
            self.list_data_dict = self.list_data_dict * repeat_time
        
        new_len = len(self.list_data_dict)
        print(f" Final length after repeat_time adjustment: {new_len}")
        

    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        try_count = 0
        # t0 = time.time()
        while True: 
            try_count += 1
            if try_count>30:
                print(f'UND loading files: try {try_count}')
            
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Unexpected list wrapping"

            if 'image' in sources[0]:  
                image_file = sources[0]['image']

                if self.root_path:
                    image_path = f"{self.root_path.rstrip('/')}/{image_file.lstrip('./').rstrip('.')}"
                else:
                    image_path = image_file.lstrip('./').rstrip('.')
                image_path = image_path.lstrip('/')

                img_bytes = self.client.get(image_path, no_cache=True)
                if img_bytes is None:
                    print(f"Warning: Failed to get image from {image_path}. Skipping item {i}.")
                    i = (i + 1) % len(self.list_data_dict) 
                    continue 

                try:
                    with Image.open(BytesIO(img_bytes)) as img: 
                        image = img.convert("RGB")
                    del img_bytes
                    break 
                except (OSError, UnidentifiedImageError, DecompressionBombError, Exception) as e:
                    print(f"Warning: Corrupt image at {image_path}, skipping item {i}. Error: {e}")
                    i = (i + 1) % len(self.list_data_dict)  
                    continue  

            else:  
                sources = [e["conversations"] for e in sources]
                image_tensor = None
                break  

        if 'image' in sources[0]:
            processor = self.data_args.image_processor
            image_tensor = processor.preprocess(image, return_tensors='pt')['pixel_values'][0].clone().detach()
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            s2=True)

        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0]
        )
        # print(image_tensor.shape)
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image_tensor
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.default_shape
            data_dict['image'] = torch.zeros(crop_size)
        data_dict['is_gen'] = False
        
        # print(f"[UND item], time={time.time() - t0:.3f}s")
        
        return data_dict


client_1 = Client('path/to/petreloss.conf')

DATASETCEPT = {
    "und_coyo": {
        "anno_folder": "path/to/coyo/anno",
        "image_folder": "path/to/coyo",
        "client_anno": client_1,
        "client_image": client_1,
        "image_key": "image",
        "prompt_key": "conversations",
        "image_process_fn": process_anno_laion_coyo,
        "prompt_process_fn": get_prompt_laion_coyo
    },
    "laion": {
        "anno_folder": "path/to/LAION-5B/anno",
        "image_folder": "path/to/LAION-5B",
        "client_anno": client_1,
        "client_image": client_1,
        "image_key": "image",
        "prompt_key": "caption",
        "image_process_fn": None,
        "prompt_process_fn": None,
    },
}


class DatasetCeph(Dataset):
    def __init__(
        self,
        tokenizer: Any, 
        image_processor: Any,
        dataset_name: Any,):
        """
        :param transform: An optional transform (e.g., torchvision transforms) applied to the PIL image.
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dataset_name = dataset_name
        
        config = DATASETCEPT[dataset_name]
        self.client_anno = config["client_anno"]
        self.client_image = config["client_image"]
        self.anno_folder = config["anno_folder"]
        self.image_folder = config["image_folder"]
        self.image_key = config["image_key"]
        self.prompt_key = config["prompt_key"]
        self.fn_image_proc = config["image_process_fn"]
        self.fn_prompt_proc = config["prompt_process_fn"]
        
        # 1. List all files in anno_folder
        if self.client_anno:
            anno_list_gen = self.client_anno.list(self.anno_folder)
            self.anno_files = sorted(list(anno_list_gen))
        elif os.path.isdir(self.anno_folder):
            self.anno_files = sorted(os.listdir(self.anno_folder))
        
        # 2. Only take limited files
        if dataset_name == 'und_coyo':
            self.anno_files = self.anno_files[:119]
            del filtered_list
        if dataset_name == 'laion':
            self.anno_files = self.anno_files[:3000]
        
        ######
        # 3. Read and accumulate all annotation records
        self.annotations = []
        for filename in self.anno_files:
            # Build the full path to the annotation file
            anno_path = f"{self.anno_folder.rstrip('/')}/{filename.lstrip('./')}"
            # print(f"loading annotation file: {anno_path}")
            data_lines = load_annotation_lines(self.client_anno, anno_path)
            self.annotations.extend(data_lines)
            del data_lines

        # Now self.annotations holds all records from the first 593 files
        self._length = len(self.annotations)
        print(f"dataset length: {self._length}")


    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        """
        Returns: (prompt, image)
        prompt: str
        image: PIL Image (transformed if self.transform is provided)
        """
        try_count = 0
        # t0 = time.time()
        
        while True: 
            try_count += 1
            if try_count>30:
                print(f'GEN {self.dataset_name } loading files: try {try_count}')
            
            record = self.annotations[idx]

            # 1. extract image path
            img_path = record.get(self.image_key, "")
            if self.fn_image_proc:
                img_path = self.fn_image_proc(img_path)
            else:
                if self.image_folder:
                    img_path = f"{self.image_folder.rstrip('/')}/{img_path.lstrip('./')}"
            
            # 2. extract prompt
            prompt = record.get(self.prompt_key, "")
            if self.fn_prompt_proc:
                prompt = self.fn_prompt_proc(prompt)

            if not prompt or is_chinese(prompt):
                idx = (idx + 1) % self._length
                continue 

            try:
                if self.client_image:
                    img_bytes = self.client_image.get(img_path, no_cache=True)
                    if img_bytes:  
                        with Image.open(BytesIO(img_bytes)) as img:
                            if img.mode == "P":
                                img = img.convert("RGBA")
                            image = img.convert("RGB")
                            if self.dataset_name == 'pixart':
                                arr = np.array(image)
                                arr = arr[..., ::-1]
                                image = Image.fromarray(arr, mode="RGB")
                        del img_bytes
                        break 
                else:
                    image = Image.open(img_path).convert('RGB')
                    break
            except (OSError, UnidentifiedImageError, DecompressionBombError, Exception) as e:
                pass  

            idx = (idx + 1) % self._length
        
        labels_text = "{}".format(prompt)
        
        # 3. format the input according to the task and training stage
        is_gen = True
        if 'und' in self.dataset_name:
            sources = [[
                {'from': 'human', 'value': '<image>\nProvide a one-sentence caption for the image.'},
                {'from': 'gpt', 'value': labels_text}]]
            data_dict = preprocess_plain(sources, self.tokenizer, s2=True)
            input_ids = data_dict["input_ids"][0]
            labels = data_dict["labels"][0]
            is_gen = False
            image = self.image_processor.preprocess(image, return_tensors='pt', gen_task=False)['pixel_values'][0]
        else:
            # For stage 2/3
            # prompt = "<|im_start|>USER: " + labels_text + " ASSISTANT: "
            # For stage 1
            prompt = labels_text
            ####################################
            # random drop out conditions for classifier-free guidance
            if random.random() < 0.1:
                # prompt = "<|video_pad|>"
                prompt = "<|im_start|>USER: <|video_pad|>. ASSISTANT: "
            ####################################
            prompt_id = self.tokenizer(prompt).input_ids
            input_ids = torch.tensor(prompt_id + [IMAGE_TOKEN_INDEX])
            labels = torch.tensor([IGNORE_INDEX]*len(prompt_id) + [IMAGE_TOKEN_INDEX])
            image = self.image_processor.preprocess(image, return_tensors='pt', gen_task=True)['pixel_values'][0]
            # print(prompt)
            # print(input_ids)
            # print(labels)

        # print(f"[GEN {self.dataset_name} item], time={time.time() - t0:.3f}s")
        
        return dict(input_ids=input_ids, labels=labels, image=image, is_gen=is_gen)
      