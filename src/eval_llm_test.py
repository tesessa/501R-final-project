# from llm_eval.api.task import ConfigurableTask

# class EmotionalMMLU(ConfigurableTask):
#     def __init__(self, emotional_prompt=""):
#         super().__init__()
#         self.emotional_prompt = emotional_prompt
    
#     def construct_prompt(self, doc):
#         system = f"{self.emotional_prompt}\n\nAnswer the following question."

#         question = doc['question']
#         choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(doc['choices'])])

#         return f"{system}\n\nQuestion: {question}\n{choices}\n\nAnswer:"

import lm_eval
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
import prompts
import json
import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Wrap for lm_eval
model = HFLM(
    pretrained=hf_model,
    tokenizer=tokenizer,
)
# model = HFLM(pretrained="meta-llama/Llama-3.1-8B-Instruct")

conditions = {
    "baseline": "",
    "excited": prompts.EXCITED_PROMPT,
    "sad": prompts.SAD_PROMPT,
    "angry": prompts.ANGRY_PROMPT,
    "neutral": prompts.NEUTRAL_PROMPT
}

results_file = "emotional_prompting_results.json"

if os.path.exists(results_file):
    print(f"Loading existing results from {results_file}")
    with open(results_file, "r") as f:
        results = json.load(f)
    print(f"Found results for: {list(results.keys())}")
else:
    print("No existing results found, starting fresh")
    results = {}

tasks = [
    "arc_easy",      # Elementary science (2,376 questions)
    "hellaswag",     # Sentence completion (10,042 questions)
    "winogrande",    # Commonsense reasoning (1,267 questions)
    "mmlu_pro_math"  # Professional math (1,000 questions)
]


for condition_name, emotional_prompt in conditions.items():
    print(f"\n=== Testing {condition_name} ===\n\n")

    try:
        result = lm_eval.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=5,
            batch_size=8,
            # Inject emotional prompt into system message
            system_instruction=emotional_prompt,
            device="cuda:0",
            use_cache='./lm_cache',
            log_samples = True,
            limit = 20
            # samples
        )

    # if condition_name == "baseline":
    #     print(f"Results for {condition_name}:\n\n {result}")
    # accs = [result['results'][task]['acc'] for task in tasks]
    # avg_acc = sum(accs) / len(accs)
    # print(f"Average Accuracy: {avg_acc:.3f}")
        results[condition_name] = result["results"]
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved for {condition_name}")
    except Exception as e:
        print(f"\nx ERROR in {condition_name}: {e}\n")
        print("Continuing to next condition...\n")
        continue
    # print(f"Accuracy: {result['results']['mmlu_pro_math']['acc']:.3f}")

# Save results
# import json
# with open("emotional_prompting_results.json", "w") as f:
#     json.dump(results, f, indent=2)