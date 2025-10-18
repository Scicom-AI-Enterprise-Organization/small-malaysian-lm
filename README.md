# small-malaysian-lm-B200

We want to compare how good Qwen3-1.7B-Base using B200 to continue pretraining on Malaysian multi-lingual corpus on different mixed precision training with proper truncated multi-packing.

## Manipulated variables

We want to compare,

- FP32 weight, BF16 activation.
- BF16 weight, BF16 activation.
- FP32 weight, All linear layers converted to TransformerEngine DelayedScaling recipe FP8 except logits, FP8 activation.
- BF16 weight, All linear layers converted to TransformerEngine DelayedScaling recipe FP8 except logits, FP8 activation.
- FP32 weight, All linear layers converted to TransformerEngine MXFP8 recipe FP8 except logits, FP8 activation.
- BF16 weight, All linear layers converted to TransformerEngine MXFP8 recipe FP8 except logits, FP8 activation.
- FP32 weight, All linear layers converted to TransformerEngine NVFP4 recipe FP4 except logits, FP8 activation.
- BF16 weight, All linear layers converted to TransformerEngine NVFP4 recipe FP4 except logits, FP8 activation.

## Why we no longer try to train on 5090

1. Linear layer provided not yet support to backward, https://github.com/NVIDIA/TransformerEngine/issues/1654
2. MXFP8 recipe not possible, https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L63
3. NVFP4 recipe not yet ready, NVIDIA/TransformerEngine#2255

## Hyperparameters

1. 131072 token batch size.
2. 100 warmup with 2e-5 warmup-stable-decay schedule.
3. Fused AdamW optimizer.
4. Because we are using B200s, we are only able to do BF16 Flash Attention 2 v2.8.3 out of the box for now.
5. https://github.com/apple/ml-cross-entropy Kahan summation FP32.
6. Single GPU, feel free to add DDP by your own.
7. Gradient checkpointing.

## How to

1. Prepare the data, run [notebook/prepare-dataset.ipynb](notebook/prepare-dataset.ipynb).

Or you can just clone,

```bash
hf download Scicom-intl/mosaic-ms-wikipedia-2023-10-01 --repo-type=dataset --local-dir=./multipacking
```

2. Run the finetuning,

- FP32 weight, BF16 activation,

```bash
bash b200-fp32-bf16.sh
```

## WanDB

We also recorded MFU and Throughput per second, WanDB project at https://wandb.ai/aies-scicom-scicom-ai/small-malaysian-lm