import json
import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from transformer_engine import pytorch as te
from transformer_engine.common import recipe
from transformers import Qwen3ForCausalLM
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from cut_cross_entropy import linear_cross_entropy
from transformers import get_linear_schedule_with_warmup
from accelerate import skip_first_batches
from tqdm import tqdm
import numpy as np
import click
import wandb

class Model(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        super_out = self.model.forward(
            input_ids = input_ids, 
            position_ids = position_ids, 
            attention_mask = attention_mask, 
            output_hidden_states = True,
            **kwargs,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            
            reduction = "sum" if num_items_in_batch is not None else "mean"
            
            loss = linear_cross_entropy(
                embeddings, 
                self.lm_head.weight, 
                labels, 
                shift=True,
                impl="cce_kahan_full_c",
                reduction=reduction,
            )
            if reduction == "sum":
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss / num_items_in_batch
            return {'loss': loss}
        return super_out

class Dataset(Dataset):
    def __init__(self, folder):
        self.dataset = LocalDataset(local=folder)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        data.pop('text', None)
        data.pop('token_type_ids', None)

        for k in data.keys():
            data[k] = data[k].astype(np.int64)
    
        return data
    
    def __len__(self):
        return len(self.dataset)

def collator(batch):
    batch = [b for b in batch if b is not None]
    input_ids = [b['input_ids'] for b in batch]
    position_ids = [b['position_ids'] for b in batch]
    labels = [b['input_ids'].copy() for b in batch]
    attention_mask = [b['attention_mask'] for b in batch]
    input_ids = np.concatenate(input_ids)
    position_ids = np.concatenate(position_ids)
    labels = np.concatenate(labels)
    query_lens = np.concatenate(attention_mask)
    cumsum = [0] + np.cumsum(query_lens).tolist()
    max_cumsum = int(np.max(cumsum))
    cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32)
    cu_seq_lens_k = torch.tensor(cumsum, dtype=torch.int32)
    max_seqlen_q = np.max(query_lens)
    return {
        'input_ids': torch.tensor(input_ids)[None],
        'position_ids': torch.tensor(position_ids)[None],
        'labels': torch.tensor(labels)[None],
        'cu_seq_lens_q': cu_seq_lens_q,
        'cu_seq_lens_k': cu_seq_lens_k,
        'max_length_q': max_seqlen_q,
        'max_length_k': max_seqlen_q
    }

@click.option('--model-name', default='Qwen3-0.6B-Base', help='Model name.')
@click.option('--torch-dtype', default='bfloat32', help='Weight type to load.')
@click.option('--fp8', is_flag=True, help='Train using FP8.')
@click.option('--train-dataset', default='mosaic', help='Train dataset folder.')
@click.option('--checkpoint-folder', default='checkpoint', help='Checkpoint folder.')
@click.option('--max-checkpoints', default=5, help='Max checkpoints to save.')
@click.option('--num-workers', default=5, help='Number of workers for dataloader.')
@click.command()
def main(model_name, torch_dtype, fp8, train_dataset, checkpoint_folder, max_checkpoints, num_workers):
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    warmup_steps = 100
    learning_rate = 2e-5
    log_interval = 1
    epoch = 1
    batch_size = 4
    grad_accumulation = 8
    save_interval = 100

    fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)
    model = Model.from_pretrained(model_name, torch_dtype=torch_dtype)
    os.makedirs(checkpoint_folder, exist_ok = True)

    train_dataset = Dataset(train_dataset)
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collator,
    )

    total_steps = epoch * len(train_loader)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    step = 1
    try:
        ckpts = sorted(glob(os.path.join(run_dir, f"checkpoint_{local_rank}_*.pt")), key=os.path.getmtime)
        ckpt = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        print(f'loaded checkpoint {ckpts[-1]}')
    except Exception as e:
        print(e)

    time.sleep(5.0)

    steps_trained_in_current_epoch = step % len(train_loader)
    train_loader = skip_first_batches(train_loader, steps_trained_in_current_epoch)
    sampler.set_epoch(step // len(train_loader))

    pbar = tqdm(total=total_steps, initial=step)
    iter_train_loader = iter(train_loader)

    wandb.init()
    while step < total_steps:
        batches = []
        for _ in range(grad_accumulation):
            try:
                batch = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(train_loader)
                batch = next(iter_train_loader)
            batches.append(batch)
        
        optim.zero_grad(set_to_none=True)
        for b in batches:
            for k in b.keys():
                b[k] = b[k].to(device, non_blocking=True)

            with autocast(dtype=torch.bfloat16, enabled=True):
                if fp8:
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        out = model(**b)
                else:
                    out = model(**b)
            
            loss = out["loss"] / grad_accumulation
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        if step % log_interval == 0:
            scalar_dict = {
                "grad_norm": grad_norm,
                "lr_g": scheduler.get_last_lr()[0],
                "loss": loss.item() * grad_accumulation,
                "global_step": step,
            }
            try:
                wandb.log(scalar_dict)
            except:
                pass
        
        if step % save_interval == 0:
            ckpt = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            }
            path = os.path.join(run_dir, f"checkpoint_{local_rank}_{step}.pt")
            torch.save(ckpt, path)
            print(f'save checkpoint {path}')
            ckpts = sorted(glob(os.path.join(run_dir, "checkpoint_*.pt")), key=os.path.getmtime)
            if len(ckpts) > max_ckpt:
                to_delete = ckpts[0]
                os.remove(to_delete)
                print(f"Deleted old checkpoint: {to_delete}")
        
        step += 1
        pbar.update(1)