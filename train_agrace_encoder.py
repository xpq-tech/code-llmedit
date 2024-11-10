import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.init as init
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from codellmeditor.util import nethook
from codellmeditor.dataset import prepare_requests, CLMEDataset
from codellmeditor.models.agrace import MLPEncoder
from pathlib import Path
import os
from tqdm import tqdm
from copy import deepcopy


def test(output1, output2, label, eps=0.5):
    euclidean_distance = F.pairwise_distance(output1, output2)
    acc_label_1 = torch.sum((label == 1) & (euclidean_distance <= eps)).item() / torch.sum((label == 1))
    acc_label_0 = torch.sum((label == 0) & (euclidean_distance > eps)).item() / torch.sum((label == 0))
    return acc_label_1, acc_label_0

# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(label.float() * euclidean_distance.pow(2) +
                                      (1 - label).float() * F.relu(self.margin - euclidean_distance).pow(2))
        return loss_contrastive

# 示例数据集
class ReprDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据加载器
def create_data_loader(data, batch_size, shuffle=True):
    all_data = []
    all_labels = []
    for item in data:
        prompt_repr = item['prompt_repr']
        rephrase_prompt_repr = item['rephrase_prompt_repr']
        loc_prompts_reprs = item['loc_prompts_reprs']
        all_data.append(torch.vstack([prompt_repr, rephrase_prompt_repr]))
        all_labels.append(1)
        for i in range(loc_prompts_reprs.shape[0]):
            all_data.append(torch.vstack([prompt_repr, loc_prompts_reprs[i]]))
            all_labels.append(0)
    all_data = torch.stack(all_data)
    all_labels = torch.tensor(all_labels)
    dataset = ReprDataset(all_data, all_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_reprs(model, tokenizer, save_file, args):
    datas = CLMEDataset(f"{args.data_dir}/{args.data_set}/{args.split}.json")
    datas = prepare_requests(datas, args.model, args.data_set)
    
    encoder_datas_mean = []
    encoder_datas_last_token = []
    for data in tqdm(datas, total=len(datas)):
        prompt = data['prompt']
        rephrase_prompt = data['rephrase_prompt']
        loc_prompts = data['specificity']['prompts']
        all_prompts = [prompt] + [rephrase_prompt] + loc_prompts
        all_toks = tokenizer(all_prompts, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad(), nethook.Trace(
                module=model,
                layer=args.inner_param,
                retain_input=True,
                retain_output=False,
                detach=True
            ) as tr:
            gen_args = {
                'input_ids': all_toks['input_ids'],
                'attention_mask': all_toks['attention_mask'],
                'max_new_tokens': 1,
                'pad_token_id': tokenizer.eos_token_id,
            }
            model.generate(**gen_args)
            x = tr.input
            loc_prompts_reprs_mean = []
            for i in range(x.size(0)):
                first_diff_pos = 0
                batch = x[i]
                for j in range(1, x.size(1)):
                    if not torch.equal(batch[j-1], batch[j]):
                        first_diff_pos = j 
                        break  
                if first_diff_pos == 1:
                    first_diff_pos = 0
                if i == 0:
                    prompt_repr_mean = x[0, first_diff_pos:,:].mean(0).cpu()
                elif i == 1:
                    rephrase_prompt_repr_mean = x[1, first_diff_pos:,:].mean(0).cpu()
                else:
                    loc_prompts_reprs_mean.append(x[i, first_diff_pos:,:].mean(0).cpu())
            encoder_datas_mean.append({
                "prompt_repr": prompt_repr_mean,
                "rephrase_prompt_repr": rephrase_prompt_repr_mean,
                "loc_prompts_reprs": torch.vstack(loc_prompts_reprs_mean)
            })
            encoder_datas_last_token.append({
                "prompt_repr": x[0, -1,:].cpu(),
                "rephrase_prompt_repr": x[1, -1,:].cpu(),
                "loc_prompts_reprs": x[2:,-1,:].cpu(),
            })

    with open(str(save_file)+"mean.pkl", 'wb') as f:
        torch.save(encoder_datas_mean, f)
    with open(str(save_file)+"last_token.pkl", 'wb') as f:
        torch.save(encoder_datas_last_token, f)
    return encoder_datas_mean

def train_encoder():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../ptms/CodeLlama-7b-hf', type=str)
    parser.add_argument('--inner_param', default='model.layers.27.mlp.down_proj', type=str)
    parser.add_argument('--data_dir', default='./data/data_loo_1', type=str)
    parser.add_argument('--data_set', default='EditConala', type=str, choices=["EditConala", "EditCodeSearchNet"])
    parser.add_argument('--split', default='train', type=str, choices=["test", "train"])
    parser.add_argument('--save_dir', default='./results/agrace_encoder_data_loo1', type=str)
    parser.add_argument('--mode', default='mean', type=str, choices=["mean", "last_token"])
    parser.add_argument('--val', default=True, type=bool)

    args = parser.parse_args()
    model = None
    if not Path(f"{args.save_dir}/{args.data_set}").exists():
        print(f"Path {args.save_dir}/{args.data_set} not exists, create it.")
        os.makedirs(f"{args.save_dir}/{args.data_set}")
    save_file_tmp = Path(f"{args.save_dir}/{args.data_set}/{args.model.split('/')[-1]}_{args.split}_")
    save_file = Path(str(save_file_tmp) + f"{args.mode}.pkl")
    encoder_train_datas = None
    if save_file.exists():
        #test load
        print(f"file {save_file} already exsits load it")
        with open(save_file, 'rb') as f:
            encoder_train_datas = torch.load(f)
            print(f"load done")
    if encoder_train_datas is None:
        model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
        tokenizer.add_bos_token = False
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # load data
        encoder_train_datas = get_reprs(model, tokenizer, save_file_tmp, args)
    
    if args.val:
        split_old = args.split
        args.split = 'test'
        val_save_file_tmp = Path(f"{args.save_dir}/{args.data_set}/{args.model.split('/')[-1]}_{args.split}_")
        val_save_file = Path(str(val_save_file_tmp) + f"{args.mode}.pkl")
        encoder_val_datas = None
        if val_save_file.exists():
            print(f"file {val_save_file} already exsits load it")
            with open(val_save_file, 'rb') as f:
                encoder_val_datas = torch.load(f)
                print(f"load done")
        if encoder_val_datas is None:
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
                tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
                tokenizer.add_bos_token = False
                tokenizer.pad_token_id = tokenizer.eos_token_id
            encoder_val_datas = get_reprs(model, tokenizer, val_save_file_tmp, args)
        args.split = split_old
     # 初始化模型、损失函数和优化器
    input_dim = encoder_train_datas[0]['prompt_repr'].shape[-1]
    output_dim = 256
    batch_size = 64
    learning_rate = 0.0001
    num_epochs = 100
    weight_decay=1e-4
    patience = 5
    best_val_acc = -1
    patience_counter = 0
    best_encoder = None

    encoder = MLPEncoder(input_dim, output_dim).cuda()
    criterion = ContrastiveLoss(margin=1)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    dataloader = create_data_loader(encoder_train_datas, batch_size)
    if args.val:
        val_data_loader = create_data_loader(encoder_val_datas, batch_size, False)
    # 训练模型 
    print("===============Starting train=================")
    for epoch in tqdm(range(num_epochs),total=num_epochs):
        i = 0
        encoder.train()
        for batch_data, batch_labels in dataloader:
            optimizer.zero_grad()
            
            data1 = batch_data[:,0,:].float().cuda()
            data2 = batch_data[:,1,:].float().cuda()
            label = batch_labels.cuda()

            output1 = encoder(data1)
            output2 = encoder(data2)

            loss = criterion(output1, output2, label)
            # if i%50 == 0:
            #     print(f'Loss: {loss.item():.4f}')
            loss.backward()
            optimizer.step()
            i += 1
            
        print("="*100)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        # Validation phase
        if args.val:
            encoder.eval()
            val_acc_1 = 0.0
            val_acc_0 = 0.0
            with torch.no_grad():
                for val_batch_data, val_batch_labels in val_data_loader:
                    val_data1 = val_batch_data[:,0,:].float().cuda()
                    val_data2 = val_batch_data[:,1,:].float().cuda()
                    val_label = val_batch_labels.cuda()

                    val_output1 = encoder(val_data1)
                    val_output2 = encoder(val_data2)

                    val_batch_acc_1, val_batch_acc_0 = test(val_output1, val_output2, val_label, eps=1)
                    val_acc_1 += val_batch_acc_1
                    val_acc_0 += val_batch_acc_0
            avg_val_acc_1 = val_acc_1 / len(val_data_loader)
            avg_val_acc_0 = val_acc_0 / len(val_data_loader)
            avg_val_acc = (avg_val_acc_1 + avg_val_acc_0) / 2
            print(f'Test after Epoch [{epoch+1}/{num_epochs}]: rephrease acc: {avg_val_acc_1:.4f}  neighbor acc: {avg_val_acc_0:.4f} avg: {avg_val_acc:.4f}')
             # Early stopping
            if avg_val_acc > best_val_acc:
                best_encoder = deepcopy(encoder)
                best_val_acc = avg_val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Save model
    if args.mode == "mean":
        save_file = Path(f"{args.save_dir}/{args.data_set}/{args.model.split('/')[-1]}_{args.inner_param}_{args.split}_eparam.pt")
    else:
        save_file = Path(f"{args.save_dir}/{args.data_set}/{args.model.split('/')[-1]}_{args.inner_param}_{args.split}_{args.mode}_eparam.pt")
    print(f"saving model to {save_file}")
    torch.save(best_encoder.state_dict(), save_file)


if __name__ == "__main__":
    train_encoder()