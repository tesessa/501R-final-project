import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from tqdm import tqdm
from datetime import datetime
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import Struct
from prompts import EMOTIONAL_PROMPTS
from run_va_classifier import VAPredictor
from empathy_model import EmpathyModel
from emotion_model import EmotionModel


def convert(obj):
    if hasattr(obj, "item"):  # tensor
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(x) for x in obj]
    else:
        return obj

def load_model(model_name):
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    
    print(f"✓ Model loaded on {model.device}") # make sure device is on CUDA
    return model, tokenizer

def generate_response(model, tokenizer, user_message, max_tokens=200, temperature=0.7):
    messages = [{"role": "user", "content": user_message}]
    
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
    va_predictor = VAPredictor()
    empathy_model = EmpathyModel()
    emotion_model = EmotionModel()
    
    print(f"\nLoading questions from {config.question_file}...")
    with open(config.question_file, "r") as f:
        questions = json.load(f)
    print(f"✓ Loaded {len(questions)} questions")
    
    for model_short_name, model_name in config.models.items():
        model, tokenizer = load_model(model_name)
        # model_short_name = model_name.split("/")[-1]

        # pass
        
        # model, tokenizer = load_model(config.model_name)
        # model_short_name = config.model_name.split("/")[-1]
        
        question_set_name = os.path.basename(config.question_file).replace(".json", "")
        # output_file = os.path.join(
        #     config.output_dir,
        #     f"responses_{model_short_name}_{question_set_name}.json"
        # )
        output_file = config.output_file.format(model_name = model_short_name, question_set_name = question_set_name)
    
        if os.path.exists(output_file):
            print(f"\n✓ Found existing results at {output_file}")
            with open(output_file, "r") as f:
                all_results = json.load(f)
            print(f"✓ Loaded {len(all_results)} existing responses")
            
            completed = {
                (r["model"], r["emotion"], r["question_id"])
                for r in all_results
            }
        else:
            all_results = []
            completed = set()
    
        total_expected = len(questions) * len(config.emotions)
        print(f"\n{'='*60}")
        print(f"Running evaluation: {len(questions)} questions × {len(config.emotions)} emotions = {total_expected} responses")
        print(f"Already completed: {len(all_results)}")
        print(f"Remaining: {total_expected - len(all_results)}")
        print(f"{'='*60}\n")


        # for emotion_name, emotion_config in EMOTIONAL_PROMPTS.items():
        #     pass
        for emotion_name in config.emotions:
            emotion_config = EMOTIONAL_PROMPTS.get(emotion_name)
            if emotion_config is None:
                print(f"Skipping emotion {emotion_name} as it doesn't exist")
                continue
                
            print(f"\n{'='*60}")
            print(f"Emotion: {emotion_name}")
            print(f"{'='*60}")
            
            for question_data in tqdm(questions, desc=f"{emotion_name}"):
                response_id = (model_short_name, emotion_name, question_data["id"])
                if response_id in completed:
                    continue
                
                user_message = emotion_config["prefix"] + question_data["question"]

                input_emotion_labels = emotion_model.predict_emotions(user_message)
                input_empathy_label = empathy_model.predict_empathy(user_message)

                response = generate_response(
                    model,
                    tokenizer,
                    user_message,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )

                output_emotion_labels = emotion_model.predict_emotions(response)
                output_empathy_labels = empathy_model.predict_empathy(response)
                output_va_label = va_predictor.predict_with_scales(response)

                print(type(output_emotion_labels))
                print(type(output_empathy_labels))

                result = {
                    "model": model_short_name,
                    "emotion": emotion_name,
                    "quadrant": emotion_config["quadrant"],
                    "valence_input": emotion_config["valence"],
                    "arousal_input": emotion_config["arousal"],
                    "emotions_input": input_emotion_labels,
                    "empathy_input": input_empathy_label[0][0],
                    "distress_input": input_empathy_label[0][1],
                    "question_id": question_data["id"],
                    "question": question_data["question"],
                    "source": question_data["source"],
                    "subject": question_data["subject"],
                    "emotional_prefix": emotion_config["prefix"],
                    "full_user_message": user_message,
                    "response": response,
                    "response_length": len(response),
                    "valence_output": output_va_label['valence_1_9'],
                    "arousal_output": output_va_label['arousal_1_9'],
                    "emotions_output": output_emotion_labels,
                    "empathy_output": output_empathy_labels[0][0],
                    "distress_output": output_empathy_labels[0][1],
                    
                    "correct_answer": question_data.get("correct_answer"),
                    "answer_choices": question_data.get("answer_choices"),
                    "correct_answer_text": question_data.get("correct_answer_text"),
                    
                    "timestamp": datetime.now().isoformat(),
                }
                
                all_results.append(result)
                completed.add(response_id)
            
            all_results = convert(all_results)
            
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✓ Saved {len(all_results)} responses to {output_file}")
        
        print(f"\n{'='*60}")
        print(f"✓ EVALUATION FOR {model_short_name} COMPLETE")
        print(f"✓ Total responses: {len(all_results)}")
        print(f"✓ Results saved to: {output_file}")
        print(f"{'='*60}")
    
    print("\n\nEVALUATION COMPLETE\n\n")
        
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
 