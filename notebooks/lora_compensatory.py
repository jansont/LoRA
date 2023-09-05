import os
import yaml
import wandb
import types
import torch
import random
import argparse
import warnings
import itertools
import warnings
import loralib as lora
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformer_lens import HookedTransformer
from gpt2_lora.data_utils import FT_Dataset
from gpt2_lora.model import GPT2LMModel, GPT2Config, LORAConfig
from gpt2_lora.training.train import train_validate
from gpt2_lora.correction_dataset import CorrectionDataset, create_lm_dataset, create_testing_dataset
import gpt2_lora.ablations as ablations
import gpt2_lora.activation_graft as activation_grafts
from gpt2_lora.training.optimizer import (
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)
from gpt2_lora.exp_utils import create_exp_dir
from gpt2_lora.training.evaluate import evaluate
from gpt2_lora.utils import set_all_trainable, set_trainable_from_graft, AverageMeter, log_experiment
from sklearn.model_selection import train_test_split


# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)
        
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def validate_args(args):
    if args.task not in ['lora_graft_finetune', 'lora_mlp_finetune', 'lora_attn_finetune', 'lora_all_finetune', 'finetune', 'graft_finetune']: 
        raise ValueError("task not recognized")
    if args.task=="lora_graft_finetune": 
        if sum([args.adapt_mlp_c_fc, args.adapt_mlp_c_proj, args.adapt_attn_c_attn, args.adapt_attn_c_proj]) == 0: 
            raise ValueError("No LoRA layers selected")
    if args.task=="lora_mlp_finetune": 
        if sum([args.adapt_mlp_c_fc, args.adapt_mlp_c_proj]) == 0: 
            raise ValueError("No LoRA MLP layers selected")
    if args.task=="lora_attn_finetune": 
        if sum([args.aadapt_attn_c_attn, args.adapt_attn_c_proj]) == 0: 
            raise ValueError("No LoRA Attention layers selected")
    if args.graft_type not in ["decomposition", "causal_total_effect", "causal_total_effect_window", "causal_direct_effect_window"]: 
        raise ValueError("graft_type not recognized")
    if args.ablation_method not in ["noise", "resample", "resample_uniform"]: 
        raise ValueError("ablation_method not recognized")
    
        
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GPT2 with LORA from Activation Grafting')
    
    # Add a new argument for the config file
    parser.add_argument('--config', default="configs/config.yaml", type=str, help='Path to the YAML config file')
    add_optimizer_params(parser)
    
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    # Set the configuration values as attributes of args
    for key, value in config.items():
        setattr(args, key, value)

    setattr(args, 'device', get_device())
    validate_args(args)
    print_args(args)
    return args

def generate_lora_configs(layer: int, n_layers: int, args : types.SimpleNamespace, adapt_attn=True):
    lora_configs = [
        {"attn" : None, "mlp" : None} for _ in range(n_layers)
    ]
    if adapt_attn:
        lora_configs[layer]["attn"] = LORAConfig(
                        layer=layer,
                        layer_type="attn",
                        adapt_attn_c_attn=args.adapt_attn_c_attn,
                        adapt_attn_c_proj=args.adapt_attn_c_proj,
                        adapt_mlp_c_fc=False,
                        adapt_mlp_c_proj=False,
                        lora_dim=args.lora_dim,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout)
    else: 
        lora_configs[layer]["mlp"] = LORAConfig(
                        layer=layer,
                        layer_type="mlp",
                        adapt_attn_c_attn=False,
                        adapt_attn_c_proj=False,
                        adapt_mlp_c_fc=args.adapt_mlp_c_fc,
                        adapt_mlp_c_proj=args.adapt_mlp_c_proj,
                        lora_dim=args.lora_dim,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout) 
    return lora_configs


def run_experiment(args): 
    
    
    
    hooked_model = HookedTransformer.from_pretrained(
            args.model_name,
            center_unembed=True,  
            center_writing_weights=True,              # Whether to center weights writing to the residual stream (ie set mean to be zero). Due to LayerNorm this doesn't change the computation.      
            fold_ln=True,                             # Whether to  fold in the LayerNorm weights to the subsequent linear layer.
            refactor_factored_attn_matrices=True,
        )
    
    
    
    
    
    
    
    
    
    
    correction_dataset = CorrectionDataset(args.fact_data)
    correction_dataloader = DataLoader(correction_dataset, batch_size=1)
    early_exit = False
    
    for batch in correction_dataloader:
        #----------------------------Prepare Correction Dataset-----------------------------#
        (prompt, subject, target, target_new, neighborhood_prompts,
                            same_attribute_prompts, training_prompts) = batch
        prompt = prompt[0] ; subject = subject[0] ; target = target[0] ; target_new = target_new[0]
        training_prompts = [prompt[0] for prompt in training_prompts]
        neighborhood_prompts = [prompt[0] for prompt in neighborhood_prompts]
        same_attribute_prompts = [prompt[0] for prompt in same_attribute_prompts]

        
        #---------------------------------Setup Model------------------------------------#
        if args.model_name == "gpt2-small":
            hf_model_name = "gpt2"
            n_layer = 12
            config = GPT2Config(
                n_embd=768, n_layer=n_layer, n_head=12, 
            )
        elif args.model_name == "gpt2-large":
            hf_model_name = args.model_name
            n_layer = 35
            config = GPT2Config(
                n_embd=1280, n_layer=n_layer, n_head=20, 
            )
        else: 
            raise ValueError("model_name not recognized")
        
        
    

    


if __name__ == '__main__':
    
    args = parse_args()
    if args.do_wandb: 
        wandb.login()
        
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)   

    run_experiment(args)
    

    # #Sample code to run multiple experiments
    # variable_of_interest = "task"
    # variable_values = ["lora_graft_finetune", "lora_mlp_finetune"]
    
    # for experiment_idx in range(len(variable_values)): 
    #     args.variable_of_interest = variable_values[experiment_idx]
    #     args.experiment_name = f"{args.experiment_name}_{args.variable_of_interest}"
    #     run_experiment(args)
    

    