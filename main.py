import os
import yaml
import wandb
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
from gpt2_lora.model import GPT2LMModel, GPT2Config
from gpt2_lora.training.train import train_validate
from gpt2_lora.correction_dataset import CorrectionDataset, create_lm_dataset
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
from timeout_decorator import timeout, TimeoutError

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
    
    all_evaluations = [] ; all_init_evaluations = []
    all_prompts = [] ; all_target = [] ; all_target_new = []
    
    for batch_idx, batch in enumerate(correction_dataloader):
        #----------------------------Prepare Correction Dataset-----------------------------#
        prompt = batch["prompt"][0]
        subject = batch["subject"][0]
        target = batch["target"][0]
        target_new = batch["target_new"][0]
        training_prompts = [p[0] for p in batch["training_prompts"]]
        reference_evaluation_prompts = [p[0] for p in batch["reference_evaluation_prompts"]]
        neighborhood_prompts = [p[0] for p in batch["neighborhood_prompts"]]
        reference_neighborhood_prompts = [p[0] for p in batch["reference_neighborhood_prompts"]]
        same_attribute_prompts = [p[0] for p in batch["same_attribute_prompts"]]
        reference_same_attribute_prompts = [p[0] for p in batch["reference_same_attribute_prompts"]]

        print(prompt)
        print(subject)
        print(target, target_new)
        
        @timeout(30)
        def timeout_resample(ablation_method):
            if ablation_method == "resample_uniform": 
                original_fact, corrupted_facts, _ = ablations.resample_ablation_uniform(hooked_model, prompt,subject,target,                                                             n_noise_samples=args.noise_samples)
            elif ablation_method=="resample":
                original_fact, corrupted_facts, _ = ablations.resample_ablation(hooked_model, prompt, subject, target, n_noise_samples=args.noise_samples, temperature=args.temperature)
            elif ablation_method=="noise": 
                original_fact, corrupted_facts, _ = ablations.noise_ablation(hooked_model, prompt,subject,target,n_noise_samples=args.noise_samples)
            else: 
                raise ValueError("ablation_method not recognized")
            return original_fact, corrupted_facts
        
        try:
            original_fact, corrupted_facts = timeout_resample(args.ablation_method)
        except TimeoutError:
            warnings.warn(f"Resample timed out for prompt {prompt}")
            continue
            
            
        all_prompts.append(original_fact) ; all_target.append(target) ; all_target_new.append(target_new)
        
        #----------------------------------Grafting--------------------------------------#
        graft_args = {        
            "model":hooked_model,
            "device":args.device,
            "clean_prompt":original_fact,
            "corrupted_prompts":corrupted_facts,
            "target":target,
            "use_mle_token_graft":args.use_mle_token_graft,
            "graft_threshold":args.graft_threshold,
        }
        
        if args.graft_type == "decomposition":
            graft = activation_grafts.DecompositionGraft(**graft_args)
        elif args.graft_type == "causal_total_effect":
            graft = activation_grafts.CausalTotalEffectGraft(**graft_args)
        elif args.graft_type == "causal_total_effect_window":
            graft_args["window_size"] = args.window_size
            graft_args["window_stride"] = args.window_stride
            graft = activation_grafts.CausalTotalEffectWindowGraft(**graft_args)
        elif args.graft_type == "causal_direct_effect_window":
            raise NotImplementedError("Causal Direct Effect Window Graft not implemented")
    
        graft.run()
        lora_configs = graft.generate_lora_configs(args)
        if len(lora_configs) == 0:
            warnings.warn("No LoRA configs generated")
        print(lora_configs)
        
        
        #--------------------------------Setup logging-----------------------------------#
        if args.do_wandb: 
            run = wandb.init(
                project=f"test_lora",
                name=f"{prompt.format(subject)} + {target} -> {target_new}",
                config=vars(args),
            )
            
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
        print(config)
        
        print("..initializing model")
        lm_net = GPT2LMModel(config, lora_configs)
        print("a")
        model = GPT2LMHeadModel.from_pretrained(hf_model_name)
        print("b")
        tokenizer = GPT2Tokenizer.from_pretrained(hf_model_name)
        print("c")
        state_dict = model.state_dict()
        print("d")
        lm_net.load_weight(state_dict)  
        
        #-----------------------------Setup Traininable Parameters---------------------------------#
        print("setting trainable parameters")
        if "lora" in args.task: 
            lora.mark_only_lora_as_trainable(lm_net)
        elif args.task=="finetune":
            set_all_trainable(lm_net)
        elif args.task=="graft_finetune":
            set_trainable_from_graft(lm_net, graft)
        else:
            raise ValueError("Task not recognized")
                
        print("creating datasets")
        if args.fp16:
            try:
                from torch.cuda import amp
            except Exception as e:
                warnings.warn('Could not import amp, apex may not be installed')
        
        if args.rank == 0:
            work_dir = os.getenv('PT_OUTPUT_DIR', 'gpt2_model')
            args.logging = create_exp_dir(work_dir)

        dataset = create_lm_dataset(
            prompts=training_prompts, target=target_new,
            subject=subject, tokenizer=tokenizer, args=args
        )
        dataset_ref = create_lm_dataset(
            prompts=training_prompts, target=target,
            subject=subject, tokenizer=tokenizer, args=args
        )
        neighbourhood_dataset = create_lm_dataset(
            prompts=neighborhood_prompts, target=target,
            subject=subject, tokenizer=tokenizer, args=args
        )
        neighbourhood_dataset_ref = create_lm_dataset(
            prompts=neighborhood_prompts, target=target_new,
            subject=subject, tokenizer=tokenizer, args=args
        )
        same_attribute_dataset = create_lm_dataset(
            prompts=same_attribute_prompts, target=target_new,
            subject=subject, tokenizer=tokenizer, args=args
        )
        same_attribute_dataset = create_lm_dataset(
            prompts=same_attribute_prompts, target=target, 
            subject=subject, tokenizer=tokenizer, args=args
        )
        same_attribute_dataset_ref = create_lm_dataset(
            prompts=reference_same_attribute_prompts, target=target_new, 
            subject=subject, tokenizer=tokenizer, args=args
        )
        dataset_indices = list(range(len(dataset)))
        training_indices, valid_indices = train_test_split(
            dataset_indices, test_size=args.test_size, random_state=args.random_seed
        )
        training_prompts = [d for i,d in enumerate(dataset) if i in training_indices]
        valid_prompts = [d for i,d in enumerate(dataset) if i in valid_indices]
        training_prompts_ref = [d for i,d in enumerate(dataset_ref) if i in training_indices]
        valid_prompts_ref = [d for i,d in enumerate(dataset_ref) if i in valid_indices]
        
        
        train_data = FT_Dataset(
            samples=training_prompts,
            ref_samples=training_prompts_ref,
            batch_size=args.train_batch_size,
            max_seq_length=args.seq_len, 
            joint_lm=args.obj=='jlm'
        ) 
        valid_data = FT_Dataset(
            samples=valid_prompts,
            ref_samples=valid_prompts_ref,
            batch_size=args.train_batch_size,
            max_seq_length=args.seq_len, 
            joint_lm=args.obj=='jlm'
        )     
        neighbourhood_data = FT_Dataset(
            samples=neighbourhood_dataset,
            ref_samples=neighbourhood_dataset_ref,
            batch_size=args.train_batch_size,
            max_seq_length=args.seq_len, 
            joint_lm=args.obj=='jlm',
        )
        same_attribute_data = FT_Dataset(
            samples=same_attribute_dataset,
            ref_samples=same_attribute_dataset_ref,
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
        neighbourhood_loader = DataLoader(
            neighbourhood_data, batch_size=len(neighbourhood_data), num_workers=0, 
            shuffle=False, pin_memory=False, drop_last=False,
        )
        same_attribute_loader = DataLoader(
            same_attribute_data, batch_size=len(same_attribute_data), num_workers=0, 
            shuffle=False, pin_memory=False, drop_last=False,
        )
        
        if args.init_checkpoint is not None:
            print('loading model pretrained weight.')
            lm_net.load_weight(torch.load(args.init_checkpoint))    
            
        if args.device=='cuda':
            print('using cuda.')
            lm_net = lm_net.cuda()

        optimizer = create_adam_optimizer_from_args(lm_net, args)
        
        print("eval")
        test_evaluation = evaluate(lm_net,valid_loader,args,tokenizer,)
        test_evaluation = {f"testing_{k}": v for k, v in test_evaluation.items()}
        #evaluating specificity
        neighbourhood_evaluation = evaluate(lm_net,neighbourhood_loader,args,tokenizer)
        neighbourhood_evaluation = {f"neighbourhood_{k}": v for k, v in neighbourhood_evaluation.items()}

        same_attribute_evaluation = evaluate(lm_net,same_attribute_loader,args,tokenizer)
        same_attribute_evaluation = {f"same_attribute_{k}": v for k, v in same_attribute_evaluation.items()}
        
        init_evaluation = {**test_evaluation, **neighbourhood_evaluation, **same_attribute_evaluation}
        
        if args.max_step is None:
            args.max_step = (args.max_epoch * train_data.num_batches) 
            print('set max_step:', args.max_step)
        print("Training")
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
            early_exit = True

        print("eval")
        test_evaluation = evaluate(lm_net,valid_loader,args,tokenizer,)
        test_evaluation = {f"testing_{k}": v for k, v in test_evaluation.items()}
        #evaluating specificity
        neighbourhood_evaluation = evaluate(lm_net,neighbourhood_loader,args,tokenizer)
        neighbourhood_evaluation = {f"neighbourhood_{k}": v for k, v in neighbourhood_evaluation.items()}

        same_attribute_evaluation = evaluate(lm_net,same_attribute_loader,args,tokenizer)
        same_attribute_evaluation = {f"same_attribute_{k}": v for k, v in same_attribute_evaluation.items()}
        
        total_evaluation = {**test_evaluation, **neighbourhood_evaluation, **same_attribute_evaluation}
            
        all_evaluations.append(total_evaluation)
        all_init_evaluations.append(init_evaluation)
        if args.do_wandb: 
            run.finish()
            
        if early_exit: 
            break
        
        if batch_idx==100: 
            break
        
        del lm_net
        del optimizer
        del scheduler
        del model
        del tokenizer
        del state_dict
        
    
    log_experiment(all_prompts, all_target, all_target_new, all_evaluations, all_init_evaluations, args)
    


if __name__ == '__main__':
    
    args = parse_args()
    key="7b93b93f91b9088eb5e2a52295c51c5d6d9fd2e3"
    if args.do_wandb: 
        wandb.login(key=key)
        
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
    

    