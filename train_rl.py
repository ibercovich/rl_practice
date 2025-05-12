#!/usr/bin/env python3
"""
Minimal REINFORCE on LeetCode, zero external RL libs.
Dependencies: torch, transformers, requests
"""

import json, pathlib, subprocess, tempfile, textwrap, requests, os, signal
from itertools import islice
import torch
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

# ---------- Model ----------
device = "cuda"
MODEL = "codellama/CodeLlama-7b-hf"
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
        ).to(device)
model.gradient_checkpointing_enable()
optim = torch.optim.AdamW(model.parameters(), lr=5e-6)

def generate(prompt, max_new=256):
    ids = tok(prompt, return_tensors="pt").to(device)
    gen = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                         temperature=0.3, pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

# ---------- Training loop (online REINFORCE) ----------
baseline = 0.0; beta = 0.9
tasks = load_tasks(DATA, split="train")

for step, task in enumerate(islice(tasks, None)):
    prompt = f"{task['prompt']}\n{task['signature']}\n```python\n"
    code   = generate(prompt)
    reward = float(run_py_solution(code, task["tests"]))

    # log‑prob of first generated token (quick & dirty credit assignment)
    ids_in = tok(prompt, return_tensors="pt").to(device)
    logits = model(**ids_in, use_cache=False).logits[:, -1, :]
    first_tok = tok(code, return_tensors="pt").input_ids[:,0].to(device)
    logp = torch.log_softmax(logits, -1).gather(1, first_tok.unsqueeze(-1)).squeeze()

    advantage = reward - baseline
    loss = -(advantage * logp)
    loss.backward(); optim.step(); optim.zero_grad()
    baseline = beta*baseline + (1-beta)*reward

    if step % 25 == 0:
        print(f"[{step}] R={reward:.0f}  baseline={baseline:.3f}")

    if step and step % 500 == 0:
        torch.save(model.state_dict(), f"ckpt_{step}.pt")