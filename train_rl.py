#!/usr/bin/env python3
"""
Minimal REINFORCE on LeetCode, zero external RL libs.
Dependencies: torch, transformers, requests
"""

import json, pathlib, subprocess, tempfile, textwrap, requests, os, signal
from itertools import islice
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- Download dataset once ----------
DS_URL = ("https://huggingface.co/datasets/newfacade/LeetCodeDataset/"
          "resolve/main/LeetCodeDataset-train.jsonl")
DATA = pathlib.Path("leetcode_train.jsonl")
if not DATA.exists():
    print("↓ downloading LeetCode split …")
    DATA.write_bytes(requests.get(DS_URL, timeout=60).content)

# ---------- Helpers ----------
def load_tasks(path, split="train"):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # No split filtering needed as the file is already the train split
            yield {
                "prompt": row["prompt"],
                "signature": row["starter_code"], # Use "starter_code"
                "tests": row["test"] # Use "test" (singular)
            }

def run_py_solution(code:str, tests:list, limit=1.0) -> bool:
    with tempfile.TemporaryDirectory() as d:
        sol = pathlib.Path(d, "solution.py");  sol.write_text(code)
        harness = textwrap.dedent(f"""
            import json, types, sys, traceback
            mod = types.ModuleType("solution")
            exec(open("{sol.name}").read(), mod.__dict__)
            fn = [v for v in mod.__dict__.values() if callable(v)][0]
            tests = json.loads(open("tests.json").read())
            try:
                for inp, exp in tests:
                    if fn(*inp) != exp:
                        sys.exit(1)
                sys.exit(0)
            except Exception:
                sys.exit(1)
        """)
        pathlib.Path(d, "runner.py").write_text(harness)
        pathlib.Path(d, "tests.json").write_text(json.dumps(tests))
        try:
            res = subprocess.run(
                ["python", "runner.py"], cwd=d,
                timeout=limit,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False

# ---------- Memory optimization configs ----------
torch.cuda.empty_cache()
gc.collect()

# Set PyTorch memory allocation config to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------- Model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use pre-quantized model from TheBloke for much lower memory usage
MODEL = "TheBloke/CodeLlama-7B-GPTQ"
REVISION = "gptq-4bit-32g-actorder_True"  # Higher quality quantization

print(f"Loading quantized model: {MODEL} (revision: {REVISION})...")
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            device_map="auto",
            trust_remote_code=False
        )

# Use optimizer with lower memory footprint
optim = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], 
    lr=5e-6, 
    weight_decay=0.01,
    eps=1e-8
)

def generate(prompt, max_new=256):
    # Free memory before generation
    torch.cuda.empty_cache()
    
    with torch.inference_mode():  # More memory efficient than no_grad
        ids = tok(prompt, return_tensors="pt").to(device)
        gen = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                            temperature=0.3, pad_token_id=tok.eos_token_id)
        return tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

# ---------- Training loop (online REINFORCE) ----------
baseline = 0.0; beta = 0.9
tasks = load_tasks(DATA, split="train")

# Gradient accumulation steps
grad_accum_steps = 8  # Accumulate gradients to save memory
accumulated_steps = 0

print("Starting training...")
for step, task in enumerate(islice(tasks, None)):
    prompt = f"{task['prompt']}\n{task['signature']}\n```python\n"
    code   = generate(prompt)
    reward = float(run_py_solution(code, task["tests"]))

    # Clear CUDA cache between operations
    torch.cuda.empty_cache()
    
    # log‑prob of first generated token (quick & dirty credit assignment)
    ids_in = tok(prompt, return_tensors="pt").to(device)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits = model(**ids_in, use_cache=False).logits[:, -1, :]
    first_tok = tok(code, return_tensors="pt").input_ids[:,0].to(device)
    logp = torch.log_softmax(logits, -1).gather(1, first_tok.unsqueeze(-1)).squeeze()

    advantage = reward - baseline
    loss = -(advantage * logp) / grad_accum_steps  # Scale the loss for accumulation
    
    # Backward pass
    loss.backward()
    
    # Only update weights after accumulating gradients
    accumulated_steps += 1
    if accumulated_steps == grad_accum_steps:
        # Gradient clipping to prevent memory spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
        accumulated_steps = 0
        
    baseline = beta*baseline + (1-beta)*reward

    # Free memory after each step
    del logits, ids_in, first_tok, logp, loss
    torch.cuda.empty_cache()
    gc.collect()

    if step % 25 == 0:
        print(f"[{step}] R={reward:.0f}  baseline={baseline:.3f}")

    if step and step % 500 == 0:
        save_path = f"ckpt_{step}.pt"
        print(f"Saving checkpoint to {save_path}...")
        torch.save(model.state_dict(), save_path)