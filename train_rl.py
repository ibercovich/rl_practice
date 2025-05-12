#!/usr/bin/env python3
"""
Minimal REINFORCE on LeetCode, zero external RL libs.
Dependencies: torch, transformers, requests, matplotlib
"""

import json, pathlib, subprocess, tempfile, textwrap, requests, os, signal
from itertools import islice
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from visualize import TrainingVisualizer
import concurrent.futures
import multiprocessing

# Avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# Define this function outside to avoid pickling issues
def run_test_batch(args):
    code, test_batch, sol_name, limit = args
    test_dir = tempfile.TemporaryDirectory()
    
    # Write solution file
    sol_path = pathlib.Path(test_dir.name, sol_name)
    sol_path.write_text(code)
    
    harness = textwrap.dedent(f"""
        import json, types, sys, traceback
        mod = types.ModuleType("solution")
        exec(open("{sol_name}").read(), mod.__dict__)
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
    
    runner_path = pathlib.Path(test_dir.name, "runner.py")
    runner_path.write_text(harness)
    
    tests_path = pathlib.Path(test_dir.name, "tests.json")
    tests_path.write_text(json.dumps(test_batch))
    
    try:
        res = subprocess.run(
            ["python", str(runner_path)], cwd=test_dir.name,
            timeout=limit,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        success = res.returncode == 0
        test_dir.cleanup()
        return success
    except subprocess.TimeoutExpired:
        test_dir.cleanup()
        return False

def run_py_solution(code:str, tests:list, limit=1.0) -> bool:
    # If there are no tests, return True
    if not tests:
        return True
    
    # Simple case: just run directly without parallelization
    if len(tests) <= 2:
        with tempfile.TemporaryDirectory() as d:
            sol_name = "solution.py"
            sol = pathlib.Path(d, sol_name)
            sol.write_text(code)
            
            harness = textwrap.dedent(f"""
                import json, types, sys, traceback
                mod = types.ModuleType("solution")
                exec(open("{sol_name}").read(), mod.__dict__)
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
    
    # For larger test sets, parallelize but without using nested functions
    try:
        sol_name = "solution.py"
        n_processes = min(multiprocessing.cpu_count(), 4)  # Use at most 4 processes
        batch_size = max(1, len(tests) // n_processes)
        test_batches = [tests[i:i+batch_size] for i in range(0, len(tests), batch_size)]
        
        # Prepare arguments for the run_test_batch function
        args_list = [(code, batch, sol_name, limit) for batch in test_batches]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = list(executor.map(run_test_batch, args_list))
            
        # All test batches must pass
        return all(results)
    except Exception as e:
        # Fall back to non-parallel execution on error
        print(f"Parallel execution failed: {e}, falling back to sequential")
        with tempfile.TemporaryDirectory() as d:
            sol = pathlib.Path(d, "solution.py")
            sol.write_text(code)
            
            harness = textwrap.dedent(f"""
                import json, types, sys, traceback
                mod = types.ModuleType("solution")
                exec(open("solution.py").read(), mod.__dict__)
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

# Keep GPTQ model but add memory-efficient optimizations
MODEL = "TheBloke/CodeLlama-7B-GPTQ"
REVISION = "gptq-4bit-32g-actorder_True"  # Higher quality quantization

print(f"Loading quantized model: {MODEL} (revision: {REVISION})...")
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")

# Enhanced loading with additional optimization flags
from transformers import BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            device_map="auto",
            trust_remote_code=False,
            # Better memory efficiency configs
            use_cache=True,  # Enable KV caching for faster generation
            attn_implementation="flash_attention_2",  # Use flash attention where applicable
        )

# Increase batch size for faster training
BATCH_SIZE = 4  # Process more examples at once (was 2)
grad_accum_steps = 2  # Reduced from 4 for faster updates

# Use optimizer with higher learning rate and mixed precision
from torch.cuda.amp import GradScaler
scaler = GradScaler()  # For mixed precision training

# Higher LR since we're using gradient scaling
optim = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], 
    lr=1e-5,  # Double the learning rate
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.95)  # Slightly higher beta2 for better stability
)

# Generate multiple candidates and pick the best one
def generate_candidates(prompt, n_candidates=3, max_new=256):
    # Free memory before generation
    torch.cuda.empty_cache()
    
    candidates = []
    with torch.inference_mode():  # More memory efficient than no_grad
        ids = tok(prompt, return_tensors="pt").to(device)
        try:
            # Generate multiple candidates
            gens = model.generate(
                **ids, 
                max_new_tokens=max_new, 
                do_sample=True,
                temperature=0.7,  
                top_p=0.95,
                top_k=50,
                num_return_sequences=n_candidates,
                num_beams=1,  # Disable beam search to allow for diversity
                pad_token_id=tok.eos_token_id
            )
            
            for i in range(gens.shape[0]):
                candidates.append(tok.decode(gens[i][ids.input_ids.shape[1]:], skip_special_tokens=True))
            return candidates
        except RuntimeError as e:
            # Fallback to a single greedy sequence if sampling fails
            print(f"Warning: Generation error: {str(e)[:100]}... Retrying with safer settings")
            torch.cuda.empty_cache()
            
            gen = model.generate(
                **ids, 
                max_new_tokens=max_new, 
                do_sample=False, 
                pad_token_id=tok.eos_token_id
            )
            return [tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)]

def generate(prompt, max_new=256):
    # Generate multiple candidates and pick the best one if possible
    candidates = generate_candidates(prompt, n_candidates=3, max_new=max_new)
    return candidates[0]  # For now, just return the first one

# ---------- Training loop (online REINFORCE) ----------
baseline = 0.0; beta = 0.9
tasks = load_tasks(DATA, split="train")

# Track training statistics
total_examples = 0
successful_examples = 0
start_time = time.time()

# Train with a larger batch size for more efficient processing
batch_size = BATCH_SIZE
batched_tasks = []

# Setup visualization
visualizer = TrainingVisualizer(window_size=100)

print("Starting training...")
for step, task in enumerate(islice(tasks, None)):
    batched_tasks.append(task)
    
    # Process tasks in small batches
    if len(batched_tasks) < batch_size and step > 0:
        continue
    
    # Process batch
    batch_rewards = []
    batch_losses = []
    batch_logps = []
    
    for batch_task in batched_tasks:
        prompt = f"{batch_task['prompt']}\n{batch_task['signature']}\n```python\n"
        
        # Generate multiple candidates and pick the one that solves the problem
        candidates = generate_candidates(prompt, n_candidates=3)
        reward = 0
        success_code = None
        
        # Try each candidate until we find one that works
        for candidate in candidates:
            candidate_reward = float(run_py_solution(candidate, batch_task["tests"]))
            if candidate_reward > 0:
                # Found a working solution!
                reward = candidate_reward
                success_code = candidate
                break
        
        # If none worked, use the first one
        if success_code is None:
            success_code = candidates[0]
        
        batch_rewards.append(reward)
        
        # Update statistics
        total_examples += 1
        successful_examples += int(reward)
        success_rate = successful_examples / total_examples
        
        # Clear CUDA cache between operations
        torch.cuda.empty_cache()
        
        # log‑prob of first generated token (quick & dirty credit assignment)
        ids_in = tok(prompt, return_tensors="pt").to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(**ids_in, use_cache=False).logits[:, -1, :]
        first_tok = tok(success_code, return_tensors="pt").input_ids[:,0].to(device)
        logp = torch.log_softmax(logits, -1).gather(1, first_tok.unsqueeze(-1)).squeeze()
        batch_logps.append(logp)

        advantage = reward - baseline
        loss = -(advantage * logp) / grad_accum_steps  # Scale the loss for accumulation
        batch_losses.append(loss)
        
        # Update visualization for each example
        visualizer.update(step, reward, baseline, loss)
        
        # Free memory after each example
        del logits, ids_in, first_tok
        torch.cuda.empty_cache()
    
    # Backward pass with mixed precision for the whole batch
    for loss in batch_losses:
        scaler.scale(loss).backward()
    
    # Update baseline with average reward
    avg_reward = sum(batch_rewards) / len(batch_rewards)
    baseline = beta*baseline + (1-beta)*avg_reward
    
    # Only update weights after accumulating gradients
    accumulated_steps += 1
    if accumulated_steps == grad_accum_steps:
        # Gradient clipping to prevent memory spikes
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step with gradient scaling
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
        accumulated_steps = 0
    
    # Clear batch
    batched_tasks = []
    
    # Free memory after processing batch
    del batch_losses, batch_logps, batch_rewards
    gc.collect()
    torch.cuda.empty_cache()

    if step % 25 == 0:
        elapsed_time = time.time() - start_time
        avg_time_per_step = elapsed_time / (step + 1) if step > 0 else 0
        print(f"[{step}] R={reward:.0f}  baseline={baseline:.3f}  success_rate={success_rate:.3f}  avg_time={avg_time_per_step:.2f}s/step")
        
        # Force plot update
        visualizer.plot()

    if step and step % 500 == 0:
        save_path = f"ckpt_{step}.pt"
        print(f"Saving checkpoint to {save_path}...")
        torch.save(model.state_dict(), save_path)
        
        # Also save the visualizer figure at checkpoint
        visualizer.save_checkpoint_plot(step)