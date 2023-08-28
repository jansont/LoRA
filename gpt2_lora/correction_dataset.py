import json
from pathlib import Path
from torch.utils.data import Dataset

def create_lm_dataset(prompts, tokenizer, args):
    tokenized_prompts = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in prompts
    ]       
    dataset = []
    for tokenized_prompt in tokenized_prompts:
        split_size = max(int(len(tokenized_prompt) * args.completion_size), 1)
        context = tokenized_prompt[:-split_size]
        completion = tokenized_prompt[-split_size:]
        dataset.append((context, completion))
    return dataset

def create_testing_dataset(prompts, tokenizer, args):
    dataset = []
    for prompt in prompts:
        prompt = prompt.split()
        context = prompt[:-1]
        completion = prompt[-1]
        dataset.append((
            tokenizer.encode(context, add_special_tokens=False), 
            tokenizer.encode(completion, add_special_tokens=False)
        )) 
    return dataset           


class CorrectionDataset(Dataset):
    def __init__(self, 
                 dataset_path: Path, 
                 use_chat_gpt: bool = False,):
        with open(dataset_path, "r") as json_file:
            loaded_data = json.load(json_file)
        self.dataset = loaded_data
        self.use_chat_gpt = use_chat_gpt
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple[str, str, str, str, list[str], list[str]]:
        sample = self.dataset[idx]
        prompt = sample["requested_rewrite"]["prompt"]
        subject = sample["requested_rewrite"]["subject"]
        target = sample["requested_rewrite"]["target_true"]["str"]
        target_new = sample["requested_rewrite"]["target_new"]["str"]
        
        neighborhood_prompts = sample["neighborhood_prompts"]
        neighborhood_prompts = [prompt.format(subject) + " " + target for prompt in neighborhood_prompts]
        
        same_attribute_prompts = sample["attribute_prompts"]
        same_attribute_prompts = [prompt.format(subject) + " " + target_new for prompt in same_attribute_prompts]
                
        try: 
            training_prompts = sample["training_prompts"]
        except KeyError: 
            raise KeyError("This dataset does not have training prompts. Please use the chat gpt dataset.")
        
        return (prompt,
                subject,
                target,
                target_new,
                neighborhood_prompts,
                same_attribute_prompts, 
                training_prompts)