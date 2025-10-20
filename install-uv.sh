curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv pip install torch==2.8.0
uv pip install transformers
uv pip install ipython ipykernel
uv pip install mosaicml-streaming
uv pip install datasets
uv pip install wandb
uv pip install pandas pyarrow
uv pip install multiprocess
uv pip install --no-build-isolation transformer_engine[pytorch]
FLASH_ATTENTION_SKIP_CUDA_BUILD="TRUE" uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install git+https://github.com/apple/ml-cross-entropy
uv pip install liger-kernel==0.6.2
ipython kernel install --user --name=small-malaysian-lm