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

os.environ["HF_DATASETS_OFFLINE"] = "1"

def load_model(model_name, device1):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        # device_map="cuda",
        device_map = "auto",
        local_files_only=True
    )

    print(f"✓ Model loaded on: {hf_model.device}")
    print(f"✓ Model dtype: {hf_model.dtype}")

    return HFLM(
        pretrained=hf_model,
        tokenizer=tokenizer,
    )


def run_eval(config):
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"\n{'='*60}")
    print(f"Loading model: {config.model_name}")
    print('='*60)

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
        print(f"Found results for: {list(results.keys())}")
    else:
        os.makedirs(os.path.dirname(config.results_file), exist_ok=True)
        results = {}

    if os.path.exists(config.samples_file):
        print(f"Loading existing samples from {config.samples_file}")
        with open(config.samples_file, "r") as f:
            samples = json.load(f)
        print(f"Found samples for: {list(samples.keys())}")
    else:
        os.makedirs(os.path.dirname(config.samples_file), exist_ok=True)
        samples = {}


    for condition_name, emotional_prompt in conditions.items():
        # if condition_name in results and config.name in results[condition_name]:
        #     print(f"SKIPPING {condition_name}")
        #     continue
        if condition_name in results:
            # Check if all tasks are completed
            all_tasks_done = all(
                task in results[condition_name] 
                for task in config.tasks
            )
            if all_tasks_done:
                print(f"\n{'='*60}")
                print(f"SKIPPING {condition_name} (already completed)")
                print('='*60)
                continue

        
        print(f"\n{'='*60}")
        print(f"Evaluating: {condition_name}")
        print('='*60)

        try:
            result = lm_eval.simple_evaluate(
                model=model,
                tasks=config.tasks,
                num_fewshot=config.num_fewshot,
                batch_size=config.batch_size,
                system_instruction=emotional_prompt,
                # device=config.device,
                limit=config.limit,
                # seed=config.seed,
                log_samples=True,
                # output_path=os.path.join(config.output_dir, condition_name),   # THIS is key
                # use_cache=config.cache_dir,
            )

            if condition_name not in results:
                results[condition_name] = {}
            
            for task in config.tasks:
                if task in result["results"]:
                    results[condition_name][task] = result["results"][task]

                    task_res = result["results"][task]
                    if 'acc' in task_res:
                        print(f" ✓ {task}: {task_res['acc']:.3f}")
                    elif 'acc_norm' in task_res:
                        print(f" ✓ {task}: {task_res['acc_norm']:.3f}")
                else:
                    print(f"  ⚠ {task}: NOT FOUND in results")

            with open(config.results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓Results for {condition_name} saved to {config.results_file}")
            
            if "samples" in result:
                if condition_name not in samples:
                    samples[condition_name] = {}
                
                for task in config.tasks:
                    if task in result["samples"]:
                        samples[condition_name][task] = result["samples"][task]
                
                # Write samples to file
                with open(config.samples_file, "w") as f:
                    json.dump(samples, f, indent=2)
                
                print(f"✓ Samples saved to {config.samples_file}")

        
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            # Save what we have
            with open(config.results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Partial results saved to {config.results_file}")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ ERROR in {condition_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save partial results anyway
            with open(config.results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print("Continuing to next condition...")
            continue

    print("\n" + "="*60)
    print("ALL CONDITIONS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    config = Struct(**config_dict)

    run_eval(config)