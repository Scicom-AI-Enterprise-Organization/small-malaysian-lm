# small-malaysian-lm-B200

We want to compare how good Qwen3-1.7B-Base using B200 to continue pretraining on Malaysian multi-lingual corpus on different mixed precision training with proper truncated multi-packing.

## Manipulated variables

We want to compare,

1. FP32 weight, BF16 activation.
2. BF16 weight, BF16 activation.
3. FP32 weight, All linear layers converted to TransformerEngine DelayedScaling recipe FP8 except logits, BF16 activation.
4. BF16 weight, All linear layers converted to TransformerEngine DelayedScaling recipe FP8 except logits, BF16 activation.
5. FP32 weight, All linear layers converted to TransformerEngine MXFP8 recipe FP8 except logits, BF16 activation.
6. BF16 weight, All linear layers converted to TransformerEngine MXFP8 recipe FP8 except logits, BF16 activation.
7. FP32 weight, All linear layers converted to TransformerEngine NVFP4 recipe FP4 except logits, BF16 activation.
8. BF16 weight, All linear layers converted to TransformerEngine NVFP4 recipe FP4 except logits, BF16 activation.
9. FP32 weight, All linear layers converted to TransformerEngine DelayedScaling recipe FP8, BF16 activation.
10. BF16 weight, All linear layers converted to TransformerEngine DelayedScaling recipe FP8, BF16 activation.
11. FP32 weight, All linear layers converted to TransformerEngine MXFP8 recipe FP8, BF16 activation.
12. BF16 weight, All linear layers converted to TransformerEngine MXFP8 recipe FP8, BF16 activation.
13. FP32 weight, All linear layers converted to TransformerEngine NVFP4 recipe FP4, BF16 activation.
14. BF16 weight, All linear layers converted to TransformerEngine NVFP4 recipe FP4, BF16 activation.

## Why we no longer try to train on 5090

1. Linear layer provided not yet support to backward, https://github.com/NVIDIA/TransformerEngine/issues/1654
2. MXFP8 recipe not possible, https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L63
3. NVFP4 recipe not yet ready, NVIDIA/TransformerEngine#2255

## Important parameters

1. Only Malaysia Malay Wikipedia with total 90955776 tokens based on Qwen3 tokenizer.
2. 131072 token batch size, one batch size is 4096 tokens, so 32 batch size to achieve 131072 tokens.
3. 100 warmup with 2e-5 warmup-stable-decay schedule.
4. Fused AdamW optimizer.
5. Because we are using B200s, we are only able to do BF16 Flash Attention 2 v2.8.3 out of the box for now, currently Flash Attention 2 is the fastest based on our benchmarked varlen causal self-attention at [Scicom-AI-Enterprise-Organization/self-attention-benchmark-B200](https://github.com/Scicom-AI-Enterprise-Organization/self-attention-benchmark-B200).
6. Liger Kernel for `swiglu`, `rms_norm` and `fused_linear_cross_entropy`, **set `rope=True` caused NaN for torch compile**.
7. Single GPU, feel free to add DDP by your own.
8. Torch compile but with some broken recompiles limit, still improved MFU.
9. 1 epoch only.

## How to

1. Prepare the data, run [notebook/prepare-dataset.ipynb](notebook/prepare-dataset.ipynb).

Or you can just clone,

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 hf download Scicom-intl/mosaic-ms-wikipedia-2023-10-01 --repo-type=dataset --local-dir=./multipacking
```

2. Run the finetuning,

- FP32 weight, BF16 activation,

```bash
bash b200-fp32-bf16.sh
```

- BF16 weight, BF16 activation,

```bash
bash b200-fp32-bf16.sh
```

- FP32 weight, All linear layers converted to TransformerEngine DelayedScaling recipe FP8 except logits, BF16 activation,

```bash
bash b200-fp32-delayedscaling-fp8.sh
```

- BF16 weight, All linear layers converted to TransformerEngine DelayedScaling recipe FP8 except logits, BF16 activation,

```bash
bash b200-bf16-delayedscaling-fp8.sh
```

- FP32 weight, All linear layers converted to TransformerEngine MXFP8 recipe FP8 except logits, BF16 activation,

```bash
bash b200-fp32-mxfp8.sh
```

- BF16 weight, All linear layers converted to TransformerEngine MXFP8 recipe FP8 except logits, BF16 activation,

```bash
bash b200-bf16-mxfp8.sh
```

- FP32 weight, All linear layers converted to TransformerEngine NVFP4 recipe FP4 except logits, BF16 activation,

```bash
bash b200-fp32-nvfp4.sh
```

- BF16 weight, All linear layers converted to TransformerEngine NVFP4 recipe FP4 except logits, BF16 activation,

```bash
bash b200-bf16-nvfp4.sh
```

## WanDB

We also recorded MFU and Throughput per second, WanDB project at https://wandb.ai/aies-scicom-scicom-ai/small-malaysian-lm

<img src="wandb.png" width="50%">

We also dumped all the records from WanDB, [wandb-dump.zip](wandb-dump.zip).

## Final results

### Throughput (tokens/sec)

| Configuration           |    Throughput |
| ----------------------- | ------------: |
| BF16–BF16               |      80,510.9 |
| FP32–NVFP4              |      80,559.8 |
| FP32–BF16               |      74,554.6 |
| BF16–NVFP4              |  **97,744.8** |
| FP32–DelayedScaling FP8 |      82,731.0 |
| BF16–DelayedScaling FP8 | **100,944.9** |
| BF16–MXFP8              |      97,625.6 |
| FP32–MXFP8              |      80,299.3 |

### MFU %

| Configuration           |   MFU (%) |
| ----------------------- | --------: |
| BF16–BF16               |     40.35 |
| FP32–NVFP4              |     40.37 |
| FP32–BF16               |     37.36 |
| BF16–NVFP4              | **48.98** |
| FP32–DelayedScaling FP8 |     41.46 |
| BF16–DelayedScaling FP8 | **50.58** |
| BF16–MXFP8              |     48.92 |
| FP32–MXFP8              |     40.24 |

### Training loss

| Configuration           |  Loss |
| ----------------------- | ----: |
| BF16–BF16               | 1.596 |
| FP32–NVFP4              | 1.274 |
| FP32–BF16               | 1.190 |
| BF16–NVFP4              | 1.684 |
| FP32–DelayedScaling FP8 | 1.194 |
| BF16–DelayedScaling FP8 | 1.542 |
| BF16–MXFP8              | 1.538 |
| FP32–MXFP8              | 1.195 |

### GPU memory utilization %

| Configuration           | Memory Utilization (%) |
| ----------------------- | ---------------------: |
| BF16–BF16               |                  53.66 |
| FP32–NVFP4              |                  62.53 |
| FP32–BF16               |                  67.91 |
| BF16–NVFP4              |              **52.44** |
| FP32–DelayedScaling FP8 |                  69.02 |
| BF16–DelayedScaling FP8 |              **56.14** |
| BF16–MXFP8              |                  58.79 |
| FP32–MXFP8              |                  67.66 |

## What we learnt

1. The BF16-weight DelayedScaling FP8 setup achieved the highest throughput (100.9k tokens/s), representing a +35% improvement over the FP32-BF16 baseline.
2. The highest MFU (50.6%) was observed under BF16-weight FP8 (DelayedScaling), not NVFP4.
3. Weirdly BF16-FP8 and BF16-FP4 got lower losses compared to BF16.
4. The BF16-weight NVFP4 run achieved the lowest memory utilization (52.4%), though at the cost of slight higher loss.

### Conclusion

1. Generally about TransformerEngine mixed precision, the low precision only happened during computation, it will upcast back later.
2. Low precision mixed precision training able to reduced memory footprint and improved the throughput.
3. Software stacks and kernel libraries (e.g., FlashAttention, Transformer Engine, cuBLAS) are still catching up, not all workloads yet fully exploit the new Blackwell potential.