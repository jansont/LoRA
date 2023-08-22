import os
import yaml
import torch
import random
import argparse
import warnings
import itertools
import loralib as lora
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from transformer_lens import HookedTransformer
from gpt2_lora.data_utils import FT_Dataset
from gpt2_lora.model import GPT2LMModel, GPT2Config
from gpt2_lora.training.train import train_validate
from gpt2_lora.correction_dataset import CorrectionDataset
from gpt2_lora.ablations import noise_ablation, resample_ablation
from gpt2_lora.activation_graft import CausalGraft, DecompositionGraft
from gpt2_lora.training.optimizer import (
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)
from gpt2_lora.exp_utils import create_exp_dir
from sklearn.model_selection import train_test_split


# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GPT2 with LORA from Activation Grafting')
    
    # Add a new argument for the config file
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    add_optimizer_params(parser)
    
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    # Set the configuration values as attributes of args
    for key, value in config.items():
        setattr(args, key, value)

    print_args(args)

    return args



if __name__ == '__main__':
    args = parse_args()

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)    

    model = HookedTransformer.from_pretrained(
            args.model_name,
            center_unembed=True,  
            center_writing_weights=True,              # Whether to center weights writing to the residual stream (ie set mean to be zero). Due to LayerNorm this doesn't change the computation.      
            fold_ln=True,                             # Whether to  fold in the LayerNorm weights to the subsequent linear layer.
            refactor_factored_attn_matrices=True,
        )
    
    correction_dataset = CorrectionDataset(args.fact_data)
    correction_dataloader = DataLoader(correction_dataset, batch_size=1)
    for batch in correction_dataloader:
        (prompt, subject, target, target_new, neighborhood_prompts,
                            same_attribute_prompts, training_prompts) = batch
        prompt = prompt[0] ; subject = subject[0] ; target = target[0] ; target_new = target_new[0]
        training_prompts = [prompt[0] for prompt in training_prompts]
        neighborhood_prompts = [prompt[0] for prompt in neighborhood_prompts]
        same_attribute_prompts = [prompt[0] for prompt in same_attribute_prompts]
        print(prompt)
        print(subject)
        print(target)
        print(target_new)
        print(training_prompts)
        print(neighborhood_prompts)
        print(same_attribute_prompts)
    
        if args.use_resample_ablation: 
            original_fact, corrupted_facts, target = resample_ablation(model, prompt,subject,target,
                                                                       n_noise_samples=args.noise_samples)
        else: 
            original_fact, corrupted_facts, target = noise_ablation(model, prompt,subject,target,
                                                                    n_noise_samples=args.noise_samples)
        
            
        graft_constructor = CausalGraft if args.use_causal_graft else DecompositionGraft
        graft = graft_constructor(
            model=model,
            clean_prompt=original_fact,
            corrupted_prompts=corrupted_facts,
            target=target,
            use_mle_token_graft=args.use_mle_token_graft,
            graft_threshold=args.graft_threshold,
        )
        graft.run()
        lora_configs = graft.generate_lora_configs(args.lora_dim, args.lora_alpha, args.lora_dropout)
        
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
                n_embd=5120, n_layer=n_layer, n_head=20, 
            )
        else: 
            raise ValueError("model_name not recognized")
        lm_net = GPT2LMModel(config, lora_configs)
        
        model = GPT2LMHeadModel.from_pretrained(hf_model_name)
        state_dict = model.state_dict()
        lm_net.load_weight(state_dict)   
        lora.mark_only_lora_as_trainable(lm_net)
                
        
        if args.fp16:
            try:
                from torch.cuda import amp
            except Exception as e:
                warnings.warn('Could not import amp, apex may not be installed')
        
        if args.rank == 0:
            work_dir = os.getenv('PT_OUTPUT_DIR', 'gpt2_model')
            args.logging = create_exp_dir(work_dir)
            
        traing_test_split = args.train_test_split
        training_prompts, valid_prompts = train_test_split(training_prompts, test_size=traing_test_split, random_state=args.random_seed)

        train_data = FT_Dataset(
            samples=training_prompts,
            batch_size=args.train_batch_size,
            max_seq_length=args.seq_len, 
            joint_lm=args.obj=='jlm'
        )     
        valid_data = FT_Dataset(
            samples=valid_prompts,
            batch_size=args.train_batch_size,
            max_seq_length=args.seq_len, 
            joint_lm=args.obj=='jlm'
        )     

        train_loader = DataLoader(
            train_data, batch_size=args.train_batch_size, num_workers=0, 
            shuffle=False, pin_memory=False, drop_last=True,
        )
        
        valid_loader = DataLoader(
            valid_data, batch_size=args.valid_batch_size, num_workers=0, 
            shuffle=False, pin_memory=False, drop_last=False,
        )
    
        if args.init_checkpoint is not None:
            print('loading model pretrained weight.')
            lm_net.load_weight(torch.load(args.init_checkpoint))    

        lm_net = lm_net.cuda()

        if args.lora_dim > 0:
            lora.mark_only_lora_as_trainable(lm_net)
        optimizer = create_adam_optimizer_from_args(lm_net, args)

        if args.max_step is None:
            args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
            print('set max_step:', args.max_step)

        scheduler = create_optimizer_scheduler(optimizer, args)
        if args.fp16:
            lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
        try:
            train_step = 0
            for epoch in itertools.count(start=1):
                train_step = train_validate(
                    lm_net, optimizer, scheduler, train_loader, valid_loader, args, 
                    train_step=train_step, epoch=epoch
                )
                
                if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                    if args.rank == 0:
                        print('-' * 100)
                        print('End of training')
                    break
        except KeyboardInterrupt:
            if args.rank == 0:
                print('-' * 100)
                print('Exiting from training early')
                
        print('cleanup dist ...')
