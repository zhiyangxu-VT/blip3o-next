#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
project_root1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
project_root2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root1)
sys.path.insert(0, project_root2)                         
from model.language_model.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM
# from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(Qwen2Config):
    model_type = "llava_qwen2"


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2Model, self).__init__(config)


class LlavaQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    # _train
    def forward_train(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images_gen: Optional[torch.FloatTensor] = None,
        images_und: Optional[torch.FloatTensor] = None,
        is_gen: Optional[bool] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs_gen, outputs_und = None, None
        
        if inputs_embeds is None:
            
            if images_und is not None:
                is_und = ~is_gen
                attention_mask_und = attention_mask[is_und]
                labels_und = labels[is_und]
                input_ids_und = input_ids[is_und]
                (
                    input_ids_und,
                    position_ids_und,
                    attention_mask_und,
                    past_key_values,
                    inputs_embeds_und,
                    labels_und,                                      # prepare_inputs_labels_for_gen
                ) = self.prepare_inputs_labels_for_multimodal(  # prepare_inputs_labels_for_multimodal
                    input_ids_und,
                    position_ids,
                    attention_mask_und,
                    past_key_values,
                    labels_und,
                    images_und,
                    image_sizes
                )
                
                input_ids_und = None
                if position_ids_und is not None: 
                    position_ids = position_ids_und
                else:
                    position_ids = None
                    
                # with self.no_sync():   
                outputs_und = super().forward(
                    input_ids = input_ids_und,
                    attention_mask = attention_mask_und,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds = inputs_embeds_und,
                    labels = labels_und,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
            
            if images_gen is not None:
                # max_len = None if (images_und==None) else torch.tensor(labels_und.shape[1], device=labels_und.device)
                max_len = None
                attention_mask_gen = attention_mask[is_gen]
                labels_gen = labels[is_gen]
                input_ids_gen = input_ids[is_gen]

                (
                    input_ids_gen,
                    position_ids_gen,
                    attention_mask_gen,
                    past_key_values,
                    inputs_embeds_gen,
                    labels_gen                                      # prepare_inputs_labels_for_gen
                ) = self.prepare_inputs_labels_for_gen(  # prepare_inputs_labels_for_multimodal
                    input_ids_gen,
                    position_ids,
                    attention_mask_gen,
                    past_key_values,
                    labels_gen,
                    images_gen,
                    image_sizes,
                    max_len,
                )
            
            # inputs_embeds = inputs_embeds_und if images_gen == None else (inputs_embeds_gen if images_und == None else torch.cat([inputs_embeds_und, inputs_embeds_gen], dim=0))
            # labels = labels_und if images_gen == None else (labels_gen if images_und == None else torch.cat([labels_und, labels_gen], dim=0))
            # attention_mask = attention_mask_und if images_gen == None else (attention_mask_gen if images_und == None else torch.cat([attention_mask_und, attention_mask_gen], dim=0))
            
                input_ids_gen = None
                if position_ids_gen is not None: 
                    position_ids = position_ids_gen
                else:
                    position_ids = None
            
                outputs_gen = super().forward_gen(
                    input_ids = input_ids_gen,
                    attention_mask = attention_mask_gen,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds = inputs_embeds_gen,
                    labels = labels_gen,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
            
            
        logits = outputs_gen.logits if outputs_gen is not None else outputs_und.logits

        loss = (
            (outputs_gen.loss + outputs_und.loss) 
            if outputs_gen is not None and outputs_und is not None
            else (outputs_gen.loss if outputs_gen is not None else outputs_und.loss)
        )

        device = attention_mask.device
        
        return CausalLMOutputWithPast(
            loss=loss.to(device) if loss is not None else None,
            logits=logits.to(device) if logits is not None else None)
     
     
    def forward_gen(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:


        return super().forward_gen(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds = inputs_embeds,
            labels = labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
         
    # _und    
    # used for inference
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_qwen2", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaQwen2ForCausalLM)
