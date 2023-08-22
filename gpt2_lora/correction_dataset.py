import json
from pathlib import Path
from torch.utils.data import Dataset

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