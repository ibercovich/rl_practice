#!/usr/bin/env python3
"""
eval_rl.py  –  Measure pass@5 on the LeetCode “test” split.

Usage:
    python eval_rl.py [checkpoint.pt]

If no checkpoint is supplied, the base CodeLlama weights are evaluated.
"""

# ---------- standard‑library ----------
import json, gzip, pathlib, subprocess, tempfile, textwrap, sys, requests, os, signal, math, random
from typing import List, Tuple

# ---------- third‑party ----------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- constants ----------
MODEL_NAME = "codellama/CodeLlama-7b-hf"
DATA_URL   = ("https://huggingface.co/datasets/newfacade/LeetCodeDataset/"
              "resolve/main/data/train-00000-of-00001-7af9c0172f3d59b9.json.gz")
DATA_PATH  = pathlib.Path("leetcode_train.json.gz")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- download dataset if missing ----------
if not DATA_PATH.exists():
    print("Downloading LeetCode split …")
    DATA_PATH.write_bytes(requests.get(DATA_URL, timeout=60).content)

# ---------- helpers ----------
def load_tasks(path: pathlib.Path, split="test"):
    """Yield dicts with prompt / signature / tests from gzip‑JSONL file."""
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == split:
                yield {
                    "prompt": row["prompt"],
                    "signature": row["signature"],
                    "tests": row["tests"],
                }

def run_py_solution(code: str, tests: List[Tuple], limit=1.0) -> bool:
    """Return True iff `code` passes *all* unit tests within `limit` seconds."""
    with tempfile.TemporaryDirectory() as d:
        sol = pathlib.Path(d, "solution.py");  sol.write_text(code)
        runner = textwrap.dedent(f"""
            import json, types, sys
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
        pathlib.Path(d, "runner.py").write_text(runner)
        pathlib.Path(d, "tests.json").write_text(json.dumps(tests))
        try:
            res = subprocess.run(
                [sys.executable, "runner.py"], cwd=d,
                timeout=limit,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False

def generate(model, tok, prompt: str, max_new=256):
    """Sample one solution from the model."""
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    gens = model.generate(
        **ids,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=0.3,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(gens[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

# ---------- load model ----------
print("Loading model …")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16 if DEVICE == "cuda" else None
).to(DEVICE)
model.eval()

# optionally load a fine‑tuned checkpoint
if len(sys.argv) > 1:
    ckpt_path = sys.argv[1]
    print(f"Loading checkpoint {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

# ---------- evaluation ----------
TEST_PROBLEMS = list(load_tasks(DATA_PATH, split="test"))  # full split (~400 problems)

def pass_at_k(model, tok, problems, k=5):
    ok = 0
    for p in problems:
        prompt = f"{p['prompt']}\n{p['signature']}\n```python\n"
        if any(run_py_solution(generate(model, tok, prompt), p["tests"]) for _ in range(k)):
            ok += 1
    return ok / len(problems)

print("Evaluating (this may take a while) …")
score = pass_at_k(model, tok, TEST_PROBLEMS, k=5)
print(f"pass@5 = {score:.2%}  on {len(TEST_PROBLEMS)} test problems")