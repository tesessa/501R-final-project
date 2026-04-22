import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import Struct
from prompts2 import EMOTIONAL_PROMPTS



def load_questions(num_questions=100):
    questions = []
    target_questions = num_questions // 4

    try:
        mmlu = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True, download_mode="reuse_cache_if_exists")

        for i, example in enumerate(mmlu):
            if i >= target_questions:
                break
            questions.append({
                "id": f'mmlu_{i}',
                "question": example["question"],
                "source": "mmlu",
                "subject": example.get("subject", "general"),
            })
    except Exception as e:
        print(f"Error loading MMLU dataset: {e}")


    try:
        truthfulqa = load_dataset("truthful_qa", "generation", split="validation", trust_remote_code=True, download_mode="reuse_cache_if_exists")
        for i, example in enumerate(truthfulqa):
            if i >= target_questions:
                break
            questions.append({
                "id": f"truthfulqa_{i}",
                "question": example["question"],
                "source": "truthfulqa",
                "subject": "truthfulness",
            })
    except Exception as e:
        print(f"Could not load TruthfulQA: {e}")

    try:
        eq_bench = load_dataset("pbevan11/EQ-Bench", split="validation", trust_remote_code=True, download_mode="reuse_cache_if_exists")
        for i, example in enumerate(eq_bench):
            if i >= target_questions:
                break
            questions.append({
                "id": f"eqbench_{i}",
                "question": example["prompt"],
                "source": "eq_bench",
                "subject": "emotional intelligence",
            })
    except Exception as e:
        print(f"Could not load EQ-Bench: {e}")
    
    
    try:
        emotionBench = load_dataset("SahandSab/EmoBench", "emotional_understanding", split="train", trust_remote_code=True, download_mode="reuse_cache_if_exists")
        for i, example in enumerate(emotionBench):
            if i >= target_questions:
                break
            question_text = f"Read this scenario and identify the emotion {example['subject']} is feeling:\n\n{example['scenario']}\n\nWhat emotion is {example['subject']} experiencing?"
            questions.append({
                "id": f"emotionbench_{i}",
                "question": question_text,
                "source": "emotionBench",
                "subject": "emotional understanding",
            })
    except Exception as e:
        print(f"Could not load EmotionBench: {e}")
    


    # try:
    #     eq_bench = load_dataset("pbevan11/EQ-Bench", split="validation", trust_remote_code=True, download_mode="reuse_cache_if_exists")
    #     for i, example in enumerate(eq_bench):
    #         if len(questions) >= (num_questions // 6) * 5:
    #             break
    #         questions.append({
    #             "id": f"eqbench_{i}",
    #             "question": example["prompt"],
    #             "source": "eq_bench",
    #             "subject": "emotional intelligence",
    #         })
    # except Exception as e:
    #     print(f"Could not load EQ-Bench: {e}")
    
    
    # try:
    #     emotionBench = load_dataset("SahandSab/EmoBench", "emotional_understanding", split="train", trust_remote_code=True, download_mode="reuse_cache_if_exists")
    #     for i, example in enumerate(emotionBench):
    #         if len(questions) >= (num_questions // 6) * 6:
    #             break
    #         questions.append({
    #             "id": f"emotionbench_{i}",
    #             "question": example["scenario"] + "\nWhat emotion out of the options would be felt?" + [emotion for emotion in example['emotion_choices']] + "\nExplain your reasoning.",
    #             "source": "emotionBench",
    #             "subject": "emotional understanding",
    #         })
    # except Exception as e:
    #     print(f"Could not load EmotionBench: {e}")


    if len(questions) < num_questions:
        fallback_questions = [
            "What is the capital of France?",
            "Explain the process of photosynthesis.",
            "What causes earthquakes?",
            "Who wrote Romeo and Juliet?",
            "How does the internet work?",
            "What is the speed of light?",
            "Explain Newton's first law of motion.",
            "What is the largest planet in our solar system?",
            "How do vaccines work?",
            "What is DNA?",
            "Explain supply and demand.",
            "What caused World War I?",
            "How does a democracy work?",
            "What is climate change?",
            "Explain how the heart works.",
            "What is artificial intelligence?",
            "How do airplanes fly?",
            "What is the water cycle?",
            "Explain the theory of evolution.",
            "What is gravity?",
        ]

        for i, q in enumerate(fallback_questions):
            if len(questions) >= num_questions:
                break
            questions.append({
                "id": f"fallback_{i}",
                "question": q,
                "source": "manual",
                "subject": "general",
            })
    
    return questions[:num_questions]

def load_model(model_name, device):
    """Load model and tokenizer"""
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer
 
def generate_response(model, tokenizer, user_message, max_tokens=200, temperature=0.7):
    """Generate response to user message"""
    
    messages = [
        {"role": "user", "content": user_message}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()
  
def run_evaluation(config):
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("Loading questions...")
    questions = load_questions(config.num_questions)
    print(f"✓ Loaded {len(questions)} questions")
    
    all_results = []
    
    for model_name in config.model_names:
        model, tokenizer = load_model(model_name, config.device)
        
        model_short_name = model_name.split("/")[-1]
        
        for emotion_name, emotion_config in EMOTIONAL_PROMPTS.items():
            print(f"\n{'='*60}")
            print(f"Model: {model_short_name} | Emotion: {emotion_name}")
            print(f"{'='*60}")
            
            for question_data in tqdm(questions, desc=f"{emotion_name}"):
                user_message = emotion_config["prefix"] + question_data["question"]
                
                response = generate_response(
                    model, 
                    tokenizer, 
                    user_message,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
                
                result = {
                    "model": model_short_name,
                    "emotion": emotion_name,
                    "quadrant": emotion_config["quadrant"],
                    "valence": emotion_config["valence"],
                    "arousal": emotion_config["arousal"],
                    "question_id": question_data["id"],
                    "question": question_data["question"],
                    "source": question_data["source"],
                    "subject": question_data["subject"],
                    "emotional_prefix": emotion_config["prefix"],
                    "full_user_message": user_message,
                    "response": response,
                    "response_length": len(response),
                    "timestamp": datetime.now().isoformat(),
                }
                
                all_results.append(result)

        output_file = os.path.join(
            config.output_dir, 
            f"responses_{model_short_name}.json"
        )
        with open(output_file, "w") as f:
            json.dump(
                [r for r in all_results if r["model"] == model_short_name], 
                f, 
                indent=2
            )
        print(f"\n✓ Saved results to {output_file}")
    
    complete_file = os.path.join(config.output_dir, "all_responses.json")
    with open(complete_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ EVALUATION COMPLETE")
    print(f"✓ Total responses: {len(all_results)}")
    print(f"✓ Results saved to: {complete_file}")
    print(f"{'='*60}")
    
    return all_results
 

if __name__ == "__main__":
    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    config = Struct(**config_dict)

    results = run_evaluation(config)
 