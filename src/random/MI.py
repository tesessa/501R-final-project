# mi_pilot.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformer_lens import HookedTransformer
import prompts

# Load model with hooks
model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device="cuda",
    dtype=torch.float16
)

# Test on ONE example
baseline_prompt = "What is 2+2? Answer:"
excited_prompt = f"{prompts.EXCITED_PROMPT}\n\nWhat is 2+2? Answer:"

# Get activations
baseline_logits, baseline_cache = model.run_with_cache(baseline_prompt)
excited_logits, excited_cache = model.run_with_cache(excited_prompt)

# Quick analysis: Which layers differ most?
for layer in range(model.cfg.n_layers):
    baseline_resid = baseline_cache[f"blocks.{layer}.hook_resid_post"]
    excited_resid = excited_cache[f"blocks.{layer}.hook_resid_post"]
    
    diff = (baseline_resid - excited_resid).norm().item()
    print(f"Layer {layer}: difference = {diff:.4f}")

# Plot
import matplotlib.pyplot as plt
differences = [...]  # Calculate for all layers
plt.plot(differences)
plt.xlabel("Layer")
plt.ylabel("Activation Difference")
plt.title("Baseline vs Excited")
plt.savefig("mi_pilot.png")