PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_PROJECT="small-malaysian-lm" \
WANDB_NAME="dummy-5090-torch-compile" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
python3 train.py \
--model-name "Qwen/Qwen3-0.6B-Base" \
--torch-dtype "bfloat16" \
--batch-size 2 \
--grad-accumulation 16 \
--device-flops "209.5e12" \
--dummy-step 20 \
--torch-compile