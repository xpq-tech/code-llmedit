import json
import typing
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
from typing import Dict, List

def prepare_requests(datas, model_name, dataset_name):
    if 'CodeQwen1.5-7B' in model_name:
        model_name = 'CodeQwen1.5-7B'
    elif 'CodeLlama-7b-hf' in model_name:
        model_name = 'CodeLlama-7b-hf'
    elif 'stable-code-3b' in model_name:
        model_name = 'stable-code-3b'
    else:
        raise NotImplementedError(f'answers of {model_name} not provided')
    if dataset_name == "EditConala":
        templete = "Intent: {}\nCode:"
        requests = [{
            'case_id': data['case_id'],
            'prompt': templete.format(data['edit_request']['intent']),
            'target_new': data['edit_request']['snippet'],
            'rephrase_prompt': templete.format(data['edit_request']['rewritten_intent']),
            'specificity': {'prompts': [templete.format(item['intent']) for item in data['neighborhoods']['items']],
                            'ground_truth':data['neighborhoods']['answers'][model_name]}
        }  for data  in datas]
    elif dataset_name == "EditCodeSearchNet":
        templete = "Code: {}\nDescription:"
        requests = [{
            'case_id': data['case_id'],
            'prompt': templete.format(data['edit_request']['code']),
            'target_new': data['edit_request']['code_doc'],
            'rephrase_prompt': templete.format(data['edit_request']['code_rewrite']),
            'specificity': {'prompts': [templete.format(item['code']) for item in data['neighborhoods']['items']],
                            'ground_truth':data['neighborhoods']['answers'][model_name]}
        }  for data  in datas]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implement!")

    return datas.from_list(requests)

def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict

class CLMEDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        size: typing.Optional[int] = None,
        tokenizer_name: str = None,
    ):  
        self.data_dir = data_dir
        if "CodeSearchNet" in data_dir:
            self.data_name = "EditCodeSearchNet"
        elif "Conala" in data_dir:
            self.data_name = "EditConala"
        else:
            raise ValueError(f"Make sure data path is correct, you current data path is {data_dir}")
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileExistsError(f"{data_dir} does not exist.")
        with open(data_dir, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]

        if tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def from_list(self, data):
        self.data = data
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

    def collate_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        rephrase = [b["rephrase_prompt"] for b in batch]

        # loc
        locs = [
                prompt 
                for b in batch
                for prompt in b["specificity"]["prompts"] 
                ]
        locs_ans = [
                prompt 
                for b in batch
                for prompt in b["specificity"]["ground_truth"] 
                ]
        src = [src_ + ' ' + trg_ for src_, trg_ in zip(src, trg)]
        rephrase = [rephrase_ + ' ' + trg_ for rephrase_, trg_ in zip(rephrase, trg)]
        locs = [loc_+ ' ' + loc_ans_ for loc_, loc_ans_ in zip(locs, locs_ans)]
        
        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "rephrase": rephrase,
            }.items()
            for k2, v2 in self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
            ).items()
        }

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"])

        edit_inner["labels"] = edit_labels

        edit_rephrase = {}
        edit_rephrase["input_ids"] = batches["rephrase_input_ids"]
        edit_rephrase["attention_mask"] = batches["rephrase_attention_mask"]
        edit_rephrase["labels"] = edit_labels

        # loc
        loc = dict(
            self.tokenizer(
                locs,
                return_tensors="pt",
                padding=True,
            )
        )

        loc_ans = dict(
            self.tokenizer(
                locs_ans,
                return_tensors="pt",
                padding=True,
            )
        )
        loc["decoder_attention_mask"] = loc_ans["attention_mask"]
        loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])


        batch = {
            "edit_inner": edit_inner,
            "edit_rephrase": edit_rephrase,
            "loc": loc,
            "raw": batch,
        }

        return dict_to(batch, "cuda:0")
    