import torch
from gpt2_lora.model import LORAConfig
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
) 
from transformer_lens.utilities import devices
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from abc import ABC, abstractmethod



class Graft(ABC):
    def __init__(
        self, 
        model: HookedTransformer,
        clean_prompt: str,
        corrupted_prompts: list[str],
        target: str,
        use_mle_token_graft: bool = False,
    ):
        self.model = model
        self.clean_prompt = clean_prompt
        self.corrupted_prompts = corrupted_prompts
        self.target = target
        self.use_mle_token_graft = use_mle_token_graft
    
    def pad_from_left(
            self, 
            tokens : torch.tensor,
            maxlen:int
        ) -> torch.tensor:
        pad_token = self.model.tokenizer.pad_token_id
        padded_tokenized_inputs = torch.zeros(tokens.shape[0], maxlen)
        
        n_pads = maxlen - tokens.shape[-1]
        padded_tokenized_inputs[:,n_pads] = pad_token
        padded_tokenized_inputs[:,n_pads:] = tokens
        return padded_tokenized_inputs.long()

    def pad_to_same_length(
            self, 
            clean_tokens: torch.tensor,
            corrupted_tokens: torch.tensor
        ) -> tuple[torch.tensor, torch.tensor]: 
        maxlen = max([clean_tokens.shape[-1], corrupted_tokens.shape[-1]])
        
        if clean_tokens.shape[-1] > corrupted_tokens.shape[-1]: 
            corrupted_tokens = self.pad_from_left(corrupted_tokens, maxlen)
        elif clean_tokens.shape[-1] < corrupted_tokens.shape[-1]: 
            clean_tokens = self.pad_from_left(clean_tokens, maxlen)
        return clean_tokens, corrupted_tokens

    def unembedding_function(
            self, residual_stack: torch.tensor, cache, mlp=False) -> float:
        #we are only interested in applying the layer norm of the final layer on the final token
        #shape: [74, 5, 10, 1280] = n_layers, prompts, tokens, d_model
        z = cache.apply_ln_to_stack(residual_stack, layer = -1, mlp_input=mlp)
        z = z @ self.model.W_U
        return z

    def generate_lora_configs(self, lora_dim, lora_alpha, lora_dropout): 
        layer_names = self.layer_names
        graft = self.graft
        lora_configs = []
        for layer_name, graft_val in zip(layer_names, graft):
            if "embed" in layer_name: 
                continue
                     
            layer = int(layer_name.split("_")[0])
            adapt_mlp = "mlp" in layer_name
            
            if graft_val: 
                config = LORAConfig(
                    layer=layer,
                    adapt_mlp=adapt_mlp,
                    lora_dim=lora_dim,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout)
                
                lora_configs.append(config)
                
        #checking the lora configs
        combinations = [
            (l.layer, l.adapt_mlp) for l in lora_configs
        ]
        if has_duplicates(combinations):
            raise ValueError("Duplicate LORA configs found")
        return lora_configs
        
class DecompositionGraft(Graft): 
    def __init__(self, 
                 model: HookedTransformer,
                 clean_prompt: str,
                 corrupted_prompts: list[str],
                 target: str, 
                 use_mle_token_graft: bool = False, 
                 graft_threshold=0.75):
        super().__init__(model, clean_prompt, corrupted_prompts, target, use_mle_token_graft)
        self.graft_threshold=graft_threshold
        
    def run(self):
        model = self.model
        
        clean_tokens = model.to_tokens(self.clean_prompt, prepend_bos=True) 
        corrupted_tokens = model.to_tokens(self.corrupted_prompts, prepend_bos=True)
        clean_tokens, corrupted_tokens = self.pad_to_same_length(clean_tokens, corrupted_tokens)
        
        target_token_idx = model.to_tokens(self.target)[:,1] 
        target_token_idx = target_token_idx.expand(corrupted_tokens.shape[0], -1)
        
        clean_logits, clean_cache = model.run_with_cache(clean_tokens, return_type="logits")
        _, corrupted_cache = model.run_with_cache(corrupted_tokens, return_type="logits")
        
        mle_token_idx = clean_logits[:,-1,:].argmax(dim=-1)
        
        token_idx = mle_token_idx if self.use_mle_token_graft else target_token_idx
        
        residual_clean_stack, layer_names = clean_cache.decompose_resid(layer=-1, return_labels=True)       
        residual_corrupted_stack = corrupted_cache.decompose_resid(layer=-1, return_labels=False)
    
        token_idx_expanded = token_idx.repeat(residual_clean_stack.shape[0],1,1)
        
        residual_clean_stack = self.unembedding_function(residual_clean_stack, clean_cache)
        residual_clean_stack = residual_clean_stack[:,:,-1,:]
        residual_clean_logits = residual_clean_stack.gather(index=token_idx_expanded, dim=-1) - residual_clean_stack.mean(dim=-1, keepdim=True)
        
        residual_corrupted_stack = self.unembedding_function(residual_corrupted_stack, corrupted_cache)
        residual_corrupted_stack = residual_corrupted_stack[:,:,-1,:]
        residual_corrupted_logits = residual_corrupted_stack.gather(index=token_idx_expanded, dim=-1) - residual_clean_stack.mean(dim=-1, keepdim=True)
        
        direct_effect = (residual_clean_logits - residual_corrupted_logits).mean(dim=-1)
        
        graft = direct_effect > (direct_effect.max() * self.graft_threshold)
        
        self.graft = graft
        self.layer_names = layer_names

        
   
    
class CausalGraft: 
    def __init__(self, 
                 model: HookedTransformer,
                 prompt: str,
                 subject: str,
                 target: str):
        pass



def has_duplicates(tuple_list):
    seen_tuples = set()

    for tuple_elem in tuple_list:
        if tuple_elem in seen_tuples:
            return True
        seen_tuples.add(tuple_elem)

    return False
