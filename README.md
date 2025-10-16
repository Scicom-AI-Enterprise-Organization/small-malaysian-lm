# small-malaysian-lm

We want to compare how good Qwen3-0.6B-Base to continue pretraining on Malaysian multi-lingual corpus on different mixed precision training with proper truncated multi-packing.

## Manipulated variables

We want to compare,

- FP32 weight, BF16 activation.
- BF16 weight, BF16 activation.
- FP32 weight, All linear layers converted to TransformerEngine FP8 except logits, FP8 activation.
- BF16 weight, All linear layers converted to TransformerEngine FP8 except logits, FP8 activation.

## Hyperparameters

1. 100k token batch size.
2. 100 warmup with 2e-5 learning rate.
3. AdamW optimizer.
4. Because we are using 5090s, we are only able to do BF16 Flash Attention 2 for now.
5. Single GPU, feel free to add DDP by your own.