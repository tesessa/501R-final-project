import lm_eval
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
import torch
import json
import os
import sys
import yaml
from datetime import datetime
import prompts
from utils import Struct

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device == "auto" else None,
    )

    return HFLM(
        pretrained=hf_model,
        tokenizer=tokenizer,
    )


def run_eval(config):
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model = load_model(config.model_name, config.device)
    conditions = {
        "baseline": "",
        "excited": prompts.EXCITED_PROMPT,
        "joyful": prompts.JOYFUL_PROMPT,
        "amused": prompts.AMUSED_PROMPT,
        "enthusiastic": prompts.ENTHUSIASTIC_PROMPT,
        "angry": prompts.ANGRY_PROMPT,
        "annoyed": prompts.ANNOYED_PROMPT,
        "afraid": prompts.AFRAID_PROMPT,
        "disgusted": prompts.DISGUSTED_PROMPT,
        "content": prompts.CONTENT_PROMPT,
        "relief": prompts.RELIEF_PROMPT,
        "satisfied": prompts.SATISFIED_PROMPT,
        "grateful": prompts.GRATEFUL_PROMPT,
        "sad": prompts.SAD_PROMPT,
        "lonely": prompts.LONELY_PROMPT,
        "bored": prompts.BORED_PROMPT,
        "fatigued": prompts.FATIGUED_PROMPT,
        "neutral": prompts.FOCUSED_PROMPT
    }           

    if os.path.exists(config.results_file):
        print(f"Loading existing results from {config.results_file}")
        with open(config.results_file, "r") as f:
            results = json.load(f)
    else:
        os.makedirs(os.path.dirname(config.results_file), exist_ok=True)
        results = {}
    
    for condition_name, emotional_prompt in conditions.items():
        print(f"\nEvaluating condition: {condition_name}")

        try:
            result = lm_eval.simple_evaluate(
                model=model,
                tasks=config.tasks,
                num_fewshot=config.num_fewshot,
                batch_size=config.batch_size,
                system_instruction=emotional_prompt,
                device=config.device,
                limit=config.limit,
                seed=config.seed,
                log_samples=True,
                output_path=os.path.join(config.output_dir, condition_name),   # THIS is key
                use_cache=config.cache_dir,
            )

            if condition_name not in results:
                results[condition_name] = {}
            
            results[condition_name][config.name] = result["results"]

            with open(config.results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Results for {condition_name} saved to {config.results_file}")
        
        except Exception as e:
            print(f"Error occurred while evaluating {condition_name}: {e}")
            continue


if __name__ == "__main__":
    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    config = Struct(**config_dict)

    run_eval(config)