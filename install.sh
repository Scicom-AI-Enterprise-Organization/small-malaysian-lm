curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv pip install torch==2.8.0
uv pip install transformers
uv pip install accelerate
uv pip install ipython ipykernel
uv pip install mosaicml-streaming
uv pip install datasets
uv pip install wandb
uv pip install pandas pyarrow
uv pip install multiprocess
FLASH_ATTENTION_SKIP_CUDA_BUILD="TRUE" uv pip install flash-attn==2.8.3 --no-build-isolation
ipython kernel install --user --name=small-malaysian-lm