import lm_eval
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
import torch
import json
import os
import sys
import yaml
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import Struct
import prompts

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True # what is this for?
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map = "auto",
        local_files_only=True
    )

    print(f"✓ Model loaded on: {hf_model.device}") # make sure model is loaded on cuda
    print(f"✓ Model dtype: {hf_model.dtype}")

    return HFLM(
        pretrained=hf_model,
        tokenizer=tokenizer,
    )


def run_eval(config):
    print(f"\n{'='*60}")
    print(f"Loading model: {config.model_name}")
    print('='*60)

    model = load_model(config.model_name)

    conditions = prompts.conditions

    emotions = config.emotions
    test_emotions = {}
    for emotion in emotions:
        if conditions.get(emotion) is not None:
            test_emotions[emotion] = conditions.get(emotion)
        else:
            print(f"Emotion {emotion} is not available for testing")
    
    # now test with test_emotions
    for task_group_name, task_list in config.task_names.items():
        print(f"\n{'='*60}")
        print(f"Running {task_group_name}")
        print('='*60)

        results_output_file = config.results_file.format(task_name = task_group_name)

        if os.path.exists(results_output_file):
            print(f"Loading results from {results_output_file}")
            with open(results_output_file, "r") as f:
                results = json.load(f)
        else:
            os.makedirs(os.path.dirname(results_output_file), exist_ok=True)
            results = {}


        if config.save_samples:
            save_samples_file = config.samples_file.format(task_name = task_group_name)
            if os.path.exists(save_samples_file):
                print(f"Loading existing samples from {save_samples_file}")
                with open(save_samples_file, "r") as f:
                    samples = json.load(f)
            else:
                os.makedirs(os.path.dirname(save_samples_file), exist_ok=True)
                samples = {}


        for condition_name, emotional_prompt in test_emotions.items():
            if condition_name in results: # need to check for task name, this is assuming all tasks are separate
                # Check if all tasks are completed
                all_tasks_done = all(
                    task in results[condition_name] 
                    for task in task_list
                )
                if all_tasks_done:
                    print(f"\n{'='*60}")
                    print(f"SKIPPING {condition_name} for {task_group_name} (already completed)")
                    print('='*60)
                    continue

            
            print(f"\n{'='*60}")
            print(f"Evaluating: {condition_name} on {task_group_name}")
            print('='*60)

            try:
                result = lm_eval.simple_evaluate(
                    model=model,
                    tasks=task_list,
                    num_fewshot=config.num_fewshot,
                    batch_size=config.batch_size,
                    system_instruction=emotional_prompt,
                    limit=config.limit,
                    log_samples=config.save_samples,
                )

                if condition_name not in results:
                    results[condition_name] = {}
                
                for task in task_list:
                    if task in result["results"]:
                        results[condition_name][task] = result["results"][task]

                        task_res = result["results"][task]
                        if 'acc' in task_res:
                            print(f" ✓ {task}: {task_res['acc']:.3f}")
                        elif 'acc_norm' in task_res:
                            print(f" ✓ {task}: {task_res['acc_norm']:.3f}")
                    else:
                        print(f"  ⚠ {task}: NOT FOUND in results")
                

                with open(results_output_file, "w") as f:
                    json.dump(results, f, indent=2)
                
                print(f"\n✓Results for {condition_name} saved to {results_output_file}")
                
                if config.save_samples and "samples" in result:
                    save_samples_file = config.samples_file.format(task_name = task_group_name)

                    if condition_name not in samples:
                        samples[condition_name] = {}
                    
                    for task in task_list: # change this
                        if task in result["samples"]:
                            samples[condition_name][task] = result["samples"][task]
                    
                    # Write samples to file
                    with open(save_samples_file, "w") as f:
                        json.dump(samples, f, indent=2)
                    
                    print(f"✓ Samples saved to {save_samples_file}")

            
            except KeyboardInterrupt:
                print("\n\n⚠ Interrupted by user")
                # Save what we have
                with open(results_output_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Partial results saved to {results_output_file}")
                sys.exit(1)
            except Exception as e:
                print(f"\n✗ ERROR in {condition_name}: {e}")
                import traceback
                traceback.print_exc()
                
                # Save partial results anyway
                with open(results_output_file, "w") as f:
                    json.dump(results, f, indent=2)
                
                print("Continuing to next condition...")
                continue

        print("\n" + "="*60)
        print(f"ALL EMOTIONS FOR {task_group_name} COMPLETE")
        print("="*60)

    print("\n" + "="*60)
    print(f"ALL EVALUATIONS COMPLETE")
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