```bash
# create keys and add github if in new machine
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
cat ~/.ssh/id_rsa.pub
# add to github
# install uv and create env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
UV_TORCH_BACKEND=auto uv pip install torch requests torchvision torchaudio  huggingface_hub transformers
export HF_HOME="$(pwd)/data/huggingface"
huggingface-cli login
python train_rl.py 
```