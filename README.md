### TODO:
- make flash attention optional since it takes time to compile
- make it easy and intuitive to visualize progress
- simplify the code as much as possible
- write bash script to install dependencies

```bash
# create keys and add github if in new machine
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" && chmod 600 ~/.ssh/id_rsa && chmod 644 ~/.ssh/id_rsa.pub && cat ~/.ssh/id_rsa.pub 
# add to github
# install uv and create env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
UV_TORCH_BACKEND=auto uv pip install torch requests torchvision torchaudio accelerate huggingface_hub optimum transformers auto-gptq matplotlib numpy bitsandbytes
uv pip install flash-attn --no-build-isolation # this is slow
export HF_HOME="$(pwd)/data/huggingface"
huggingface-cli login
git config --global user.email "ibercovich@gmail.com"
git config --global user.name "Ivan Bercovich"
python train_rl.py 
```
