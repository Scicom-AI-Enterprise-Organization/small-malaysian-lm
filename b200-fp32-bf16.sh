PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_LOGS=recompiles \
WANDB_PROJECT="small-malaysian-lm" \
WANDB_NAME="b200-fp32-bf16" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
python3 train.py \
--model-name "Qwen/Qwen3-1.7B-Base" \
--torch-dtype "float32" \
--batch-size 8 \
--grad-accumulation 4 \
--num-workers 10 \
--torch-compile