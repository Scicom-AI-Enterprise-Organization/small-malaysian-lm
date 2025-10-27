PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_PROJECT="small-malaysian-lm" \
WANDB_NAME="b200-fp32-all-mxfp8" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
python3 train.py \
--model-name "Qwen/Qwen3-1.7B-Base" \
--torch-dtype "float32" \
--fp8-recipe "mxfp8" \
--include-lm-head \
--include-layernorm \
--include-rmsnorm \
--batch-size 8 \
--grad-accumulation 4 \
--num-workers 10 \
--torch-compile