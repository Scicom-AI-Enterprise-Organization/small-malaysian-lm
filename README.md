# small-malaysian-lm

We want to compare how good Qwen3-0.6B-Base to continue pretraining on Malaysian multi-lingual corpus on different mixed precision training with proper truncated multi-packing.

## Manipulated variables

We want to compare,

- FP32 weight, BF16 activation.
- BF16 weight, BF16 activation.
- FP32 weight, All linear layers converted to TransformerEngine DelayedScaling recipe FP8 except logits, FP8 activation.
- BF16 weight, All linear layers converted to TransformerEngine DelayedScaling recipe FP8 except logits, FP8 activation.

### Things not possible to do in 5090 yet

- MXFP8 recipe not possible, https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py#L63 
- NVFP4 recipe not yet ready, https://github.com/NVIDIA/TransformerEngine/issues/2255

## Hyperparameters

1. 100k token batch size.
2. 100 warmup with 2e-5 learning rate.
3. AdamW optimizer.
4. Because we are using 5090s, we are only able to do BF16 Flash Attention 2 v2.8.3 out of the box for now.
5. Single GPU, feel free to add DDP by your own.