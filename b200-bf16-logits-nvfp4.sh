PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_LOGS=recompiles \
WANDB_PROJECT="small-malaysian-lm" \
WANDB_NAME="b200-bf16-logits-nvfp4" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
python3 train.py \
--model-name "Qwen/Qwen3-1.7B-Base" \
--torch-dtype "bfloat16" \
--fp8-recipe "nvfp4" \
--include-lm-head \
--batch-size 8 \
--grad-accumulation 4 \
--num-workers 10 \
--torch-compile