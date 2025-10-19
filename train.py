import torch

torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.allow_unspec_int_on_nn_module = True

import json
import os
import time
from glob import glob
from types import MethodType
from contextlib import nullcontext
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformer_engine import pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy
from transformers import (
    set_seed,
    get_wsd_schedule,
    Qwen3ForCausalLM,
)
from accelerate import skip_first_batches
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from cut_cross_entropy import linear_cross_entropy
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
from tqdm import tqdm
import numpy as np
import click
import wandb

apply_liger_kernel_to_qwen3(
    rope=True,
    swiglu=True,
    rms_norm=True,
    cross_entropy=False,
    fused_linear_cross_entropy=True,
)

def convert_model(model):
    """
    Recursively converts the linear and layernorm layers of a model to their `transformers_engine` counterpart.
    Modified from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/transformer_engine.py#L26
    Should apply after load the model with intended precision.
    """

    for name, module in model.named_children():
        if "lm_head" in name:
            continue
        if isinstance(module, nn.Linear):
            has_bias = module.bias is not None
            params_to_gather = [module.weight]
            if any(p % 16 != 0 for p in module.weight.shape):
                return
            te_module = te.Linear(
                module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
            )
            te_module.weight.copy_(module.weight)
            if has_bias:
                te_module.bias.copy_(module.bias)

            setattr(model, name, te_module)
        else:
            convert_model(module)

class Model(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def estimate_flops(self, t = 4096):
        """
        Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 
        Borrow from https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L220
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.model.embed_tokens.weight.numel()
        l, h, q = self.config.num_hidden_layers, self.config.num_attention_heads, self.config.head_dim
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

class UInt32(Encoding):
    def encode(self, obj) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes):
        return np.frombuffer(data, np.uint32)

_encodings['uint32'] = UInt32

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
    max_seqlen_q = int(np.max(query_lens))
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
@click.option('--torch-dtype', default='bfloat16', help='Weight type to load.')
@click.option('--fp8-recipe', default=None, help='FP8 recipe.')
@click.option('--torch-compile', is_flag=True, help='Torch compile.')
@click.option('--train-dataset', default='multipacking', help='Train dataset folder.')
@click.option('--checkpoint-folder', default='checkpoint', help='Checkpoint folder.')
@click.option('--max-checkpoints', default=5, help='Max checkpoints to save.')
@click.option('--num-workers', default=5, help='Number of workers for dataloader.')
@click.option('--batch-size', default=5, help='batch size.')
@click.option('--grad-accumulation', default=4, help='gradient accumulation.')
@click.option('--device-flops', default='2.25e15', help='device flops.')
@click.option('--dummy-step', default=None, help='dummy steps.')
@click.option('--torch-profiling', is_flag=True, help='Profile using PyTorch profiling.')
@click.command()
def main(
    model_name, 
    torch_dtype, 
    fp8_recipe, 
    torch_compile, 
    train_dataset, 
    checkpoint_folder, 
    max_checkpoints, 
    num_workers,
    batch_size,
    grad_accumulation,
    device_flops,
    dummy_step,
    torch_profiling,
):
    device = torch.device("cuda", 0)
    os.makedirs(checkpoint_folder, exist_ok = True)
    device_flops = float(device_flops)

    set_seed(42)
    
    warmup_steps = 100
    learning_rate = 2e-5
    log_interval = 1
    epoch = 1
    save_interval = 1000
    max_ckpt = 5

    if torch_profiling:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
        )
    else:
        profiler = nullcontext()
    
    if fp8_recipe == 'delayedscaling':
        fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)
    elif fp8_recipe == 'mxfp8':
        fp8_recipe = recipe.MXFP8BlockScaling()
    elif fp8_recipe == 'nvfp4':
        fp8_recipe = recipe.NVFP4BlockScaling()
    elif fp8_recipe is None:
        pass
    else:
        raise ValueError('FP8 recipe not supported.')

    model = Model.from_pretrained(
        model_name, 
        attn_implementation='flash_attention_2',
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False
    if fp8_recipe is not None:
        with torch.no_grad():
            convert_model(model)

    def autocast_forward(model_forward):
        def forward(self, *args, **kwargs):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                if fp8_recipe is not None:
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        return model_forward(*args, **kwargs)
                else:
                    return model_forward(*args, **kwargs)
        forward.__wrapped__ = model_forward
        return forward
    
    new_forward = autocast_forward(model.forward)
    if hasattr(model.forward, "__func__"):
        model.forward = MethodType(new_forward, model)
    else:
        model.forward = new_forward

    _ = model.to(device)
    if torch_compile:
        model = torch.compile(model)

    train_dataset = Dataset(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=10,
        pin_memory=True,
        collate_fn=collator,
    )

    total_steps = epoch * len(train_loader)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=True)
    scheduler = get_wsd_schedule(optim, warmup_steps, int(total_steps * 0.2), num_training_steps=total_steps)

    step = 0
    try:
        ckpts = sorted(glob(os.path.join(checkpoint_folder, f"checkpoint_*.pt")), key=os.path.getmtime)
        ckpt = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        print(f'loaded checkpoint {ckpts[-1]}')
    except Exception as e:
        print(e)

    steps_trained_in_current_epoch = step % len(train_loader)
    train_loader = skip_first_batches(train_loader, steps_trained_in_current_epoch)

    pbar = tqdm(total=total_steps, initial=step)
    iter_train_loader = iter(train_loader)

    num_flops_per_token = model.estimate_flops()

    wandb.init()
    with profiler as prof:
        while step < total_steps:
            batches = []
            for _ in range(grad_accumulation):
                try:
                    batch = next(iter_train_loader)
                except StopIteration:
                    iter_train_loader = iter(train_loader)
                    batch = next(iter_train_loader)
                batches.append(batch)
            
            torch.cuda.synchronize()
            t0 = time.time()
            
            for b in batches:
                for k in b.keys():
                    if isinstance(b[k], torch.Tensor):
                        b[k] = b[k].to(device, non_blocking=True)
                    
                out = model(**b, use_cache=False)
                loss = out["loss"] / grad_accumulation
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0

            throughput_per_sec = len(batches) * batch_size * 4096 / dt
            flops_per_sec = num_flops_per_token * throughput_per_sec
            mfu = 100 * flops_per_sec / device_flops

            if (step + 1) % log_interval == 0:
                scalar_dict = {
                    "grad_norm": grad_norm,
                    "lr_g": scheduler.get_last_lr()[0],
                    "loss": loss.item() * grad_accumulation,
                    "global_step": step,
                    "mfu": mfu,
                    "throughput_per_sec": throughput_per_sec,
                }
                try:
                    wandb.log(scalar_dict)
                except:
                    pass
            
            if (step + 1) % save_interval == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                }
                path = os.path.join(checkpoint_folder, f"checkpoint_{step}.pt")
                torch.save(ckpt, path)
                print(f'save checkpoint {path}')
                ckpts = sorted(glob(os.path.join(checkpoint_folder, "checkpoint_*.pt")), key=os.path.getmtime)
                if len(ckpts) > max_ckpt:
                    to_delete = ckpts[0]
                    os.remove(to_delete)
                    print(f"Deleted old checkpoint: {to_delete}")
            
            step += 1
            pbar.update(1)

            if dummy_step is not None:
                if step >= int(dummy_step):
                    break
    
    if torch_profiling:
        prof.export_chrome_trace(f'profiling.json')

if __name__ == '__main__':
    main()