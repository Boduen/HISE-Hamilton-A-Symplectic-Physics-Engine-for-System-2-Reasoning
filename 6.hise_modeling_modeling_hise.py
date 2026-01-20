import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers import PreTrainedModel, CausalLMOutputWithCrossAttentions
from transformers.modeling_outputs import BaseModelOutputWithPast


from ..config import HISEConfig
from .base_layers import SoftTCMLayer


class HISEPreTrainedModel(PreTrainedModel):
    config_class = HISEConfig
    base_model_prefix = "hise"
    _no_split_modules = ["SoftTCMLayer"] 


    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HISEModel(HISEPreTrainedModel):
    def __init__(self, config: HISEConfig):
        super().__init__(config)
        self.embed_dim = config.d_model
        
        # 1. Embeddings
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        
        # 2. Physics Layers (Stack of Soft-TCM with Cognitive Gearbox)
        self.layers = nn.ModuleList([
            SoftTCMLayer(config) for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_momentums: Optional[List[torch.FloatTensor]] = None, 
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_fsi: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPast, Tuple]:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_fsi = output_fsi if output_fsi is not None else False
        
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        device = input_ids.device
        
        # Position Embeddings
        if past_momentums is not None:
            past_length = past_momentums[0].shape[1]
        else:
            past_length = 0
            
        pos = torch.arange(past_length, past_length + input_shape[-1], dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).view(-1, input_shape[-1])
        
        hidden_states = self.wte(input_ids) + self.wpe(pos)
        
        # Causal Mask
        if attention_mask is None:
            attention_mask = torch.triu(torch.ones(input_shape[-1], input_shape[-1], device=device) * float('-inf'), diagonal=1)


        next_momentums = []
        all_fsi_scores = [] # System 2 Monitoring Data
        
        for i, layer in enumerate(self.layers):
            layer_past = past_momentums[i] if past_momentums is not None else None
            
            # Physics Step: System 1/2 Dynamics happen here
            # Returns: hidden_states, m_new (Momentum), fsi (Risk Metric)
            hidden_states, m_new, fsi = layer(
                hidden_states, 
                mask=attention_mask, 
                past_momentum=layer_past
            )
            
            if use_cache:
                next_momentums.append(m_new)
                
            if output_fsi:
                all_fsi_scores.append(fsi)


        hidden_states = self.ln_f(hidden_states)


        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_momentums if use_cache else None,
            # We return raw FSI scores in hidden_states field when output_fsi is True
            # This allows the HISEForCausalLM head to process them
            hidden_states=tuple(all_fsi_scores) if output_fsi else None, 
        )


class HISEForCausalLM(HISEPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]


    def __init__(self, config):
        super().__init__(config)
        self.model = HISEModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_momentums: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_fsi: Optional[bool] = False,
    ) -> CausalLMOutputWithCrossAttentions:
        
        outputs = self.model(
            input_ids,
            past_momentums=past_momentums,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_fsi=output_fsi
        )
        
        hidden_states = outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)


        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        # S-Tier Feature: Aggregate FSI for Safety Valve
        # We overload the 'attentions' field to carry the FSI metric
        fsi_metric = None
        if output_fsi and outputs.hidden_states is not None:
             # Stack layers: [Layers, Batch, Seq]
             fsi_stack = torch.stack(outputs.hidden_states)
             # Average across layers to get a global System 2 risk score per token
             fsi_metric = fsi_stack.mean(dim=0)


        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, 
            attentions=fsi_metric, # <--- FSI Data exposed here
        )