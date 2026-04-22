# this was the conflicting emotions task I was trying to run lol, didn't run completely unfortunately (will have to rerun it, but we'll do one trial to make sure it works and save each prompt)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, AutoConfig
import json
import os
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn
import numpy as np
import yaml
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompts import CONFLICT_PAIRS
from empathy_model import EmpathyModel
from emotion_model import EmotionModel
from run_va_classifier import VAPredictor




# CONFIG = {
#     "models": [
#         "meta-llama/Llama-3.1-8B-Instruct",
#     ],
#     "max_tokens": 300,
#     "temperature": 0.7,
#     "output_dir": "results/conflicting_emotions",
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
# }

# Emotion labels
# emotion_labels = [
#     'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
#     'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
#     'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
#     'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
#     'relief', 'remorse', 'sadness', 'surprise', 'neutral'
# ]

def convert(obj):
    if hasattr(obj, "item"):  # tensor
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(x) for x in obj]
    else:
        return obj

def generate_with_activations(model, tokenizer, user_message, max_tokens=300, temperature=0.7):
    messages = [
        {"role": "system", "content": "Try to respond to the instructions in less than 200 words."},
        {"role": "user", "content": user_message}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    inputs = {
        "input_ids": input_ids
    }

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True
        )

        generated = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        generated[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return {
        "response": response.strip(),
        "hidden_states": tuple(h.cpu() for h in outputs.hidden_states),
        "attentions": tuple(a.cpu() for a in outputs.attentions),
        "input_ids": input_ids
    }


def compare_activations(hidden_conflict, hidden_control):
    layer_diffs = []

    for layer_idx in range(len(hidden_conflict)):
        h1 = hidden_conflict[layer_idx]
        h2 = hidden_control[layer_idx]

        # match sequence length (truncate to min)
        # min_len = min(h1.shape[1], h2.shape[1])
        # h1 = h1[:, :min_len, :]
        # h2 = h2[:, :min_len, :]
        N = min(h1.shape[1], h2.shape[1], 50)

        h1 = h1[:, -N:, :]
        h2 = h2[:, -N:, :]

        # compute metrics
        diff = h1 - h2
        l2 = torch.norm(diff).item()
        # cosine = F.cosine_similarity(h1.flatten(), h2.flatten(), dim=0).item()
        cosine = F.cosine_similarity(
            h1.view(-1, h1.shape[-1]),
            h2.view(-1, h2.shape[-1]),
            dim=-1
        ).mean().item()

        layer_diffs.append({
            "layer": layer_idx,
            "l2_diff": l2,
            "cosine_sim": cosine
        })

    return layer_diffs


def patch_layer_and_generate(model, tokenizer, inputs, control_hidden, layer_to_patch, max_tokens=100):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (control_hidden[layer_to_patch],) + output[1:]
        return control_hidden[layer_to_patch]

    handle = model.model.layers[layer_to_patch].register_forward_hook(hook_fn)

    with torch.no_grad():
        generated = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    handle.remove()

    return tokenizer.decode(generated[0], skip_special_tokens=True)



# predictor = VAPredictor(model_dir, use_cuda=False)


# def predict_emotions(text, threshold=0.4):
#     emotion_labels = [
#         'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
#         'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
#         'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
#         'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
#         'relief', 'remorse', 'sadness', 'surprise', 'neutral'
#     ]
#     tokenizer = AutoTokenizer.from_pretrained("duelker/samo-goemotions-deberta-v3-large", use_fast=False)
#     model = AutoModelForSequenceClassification.from_pretrained("duelker/samo-goemotions-deberta-v3-large")  
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    
#     with torch.no_grad():
#         logits = model(**inputs).logits
#         probabilities = torch.sigmoid(logits).squeeze(0)
    
#     predictions = {}
#     for i, emotion in enumerate(emotion_labels):
#         predictions[emotion] = {
#             'probability': float(probabilities[i]),
#             'predicted': probabilities[i] > threshold
#         }
    
#     return predictions



# def predict_empathy(texts):
#     model_name = "bdotloh/roberta-base-empathy"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     model.eval()
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits

#     #probabilities = torch.sigmoid(logits).squeeze(0)
#     scores = logits.tolist()
#     return scores   

def load_model(model_name):
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    
    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer

def generate_response(model, tokenizer, user_message, max_tokens=300, temperature=0.7):
    messages = [
        {"role": "system", "content": "Try to respond to the instructions in less than 200 words."},
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


def run_conflict_evaluation(config):
    os.makedirs(config.output_dir, exist_ok=True)

    empathy_model = EmpathyModel()
    emotion_model = EmotionModel()
    va_classifier = VAPredictor()
    
    all_results = []
    
    for model_name in config.models:
        model, tokenizer = load_model(model_name)
        
        model_short_name = model_name.split("/")[-1]
        
        print(f"\n{'='*60}")
        print(f"Running {len(CONFLICT_PAIRS)} conflict pairs on {model_short_name}")
        print(f"{'='*60}")
        
        # For each conflict pair
        for pair in tqdm(CONFLICT_PAIRS, desc="Conflict pairs"):
            # Construct full prompt
            user_message = f"{pair['emotion']}\n\n {pair['task']}"
            user_task_alone = pair['task']
            
            # Generate response
            # response = generate_response(
            #     model,
            #     tokenizer,
            #     user_message,
            #     max_tokens=config["max_tokens"],
            #     temperature=config["temperature"],
            # )

            # response_2 = generate_response(
            #     model,
            #     tokenizer,
            #     user_task_alone,
            #     max_tokens=config["max_tokens"],
            #     temperature=config["temperature"]
            # )
            out_conflict = generate_with_activations(model, tokenizer, user_message)
            out_control = generate_with_activations(model, tokenizer, user_task_alone)

            response = out_conflict["response"]
            response_2 = out_control["response"]

            # 🔍 Compare activations
            layer_diffs = compare_activations(
                out_conflict["hidden_states"],
                out_control["hidden_states"]
            )
            
            emotions_1 = emotion_model.predict_emotions(response)

            emotions_2 = emotion_model.predict_emotions(response_2)

            emotions_input_1 = emotion_model.predict_emotions(user_message)
            emotions_input_2 = emotion_model.predict_emotions(user_task_alone)

            empathy_input_1 = empathy_model.predict_empathy(user_message)
            empathy_input_2 = empathy_model.predict_empathy(user_task_alone)

            empathy_output_1 = empathy_model.predict_empathy(response)
            empathy_output_2 = empathy_model.predict_empathy(response_2)

            input_val_ars1 = va_classifier.predict_with_scales(user_message)
            input_val_ars2 = va_classifier.predict_with_scales(user_task_alone)
            output_val_ars1 = va_classifier.predict_with_scales(response)
            output_val_ars2 = va_classifier.predict_with_scales(response_2)
            # Save result
            result = {
                "model": model_short_name,
                "conflict_id": pair["id"],
                "conflict_type": pair["conflict_type"],
                "emotion": pair["emotion"],
                "emotion_type": pair["emotion_type"],
                "emotion_valence": pair.get("emotion_valence"),
                "emotion_arousal": pair.get("arousal"),
                "task": pair["task"],
                "expected_tone": pair["expected_tone"],
                "task_valence": pair.get("task_valence"),
                "task_arousal": pair.get("task_arousal"),
                "full_prompt": user_message,
                "response": response,
                "response_length": len(response),
                "timestamp": datetime.now().isoformat(),
                "emotions_input": emotions_input_1,
                "emotions_output": emotions_1,
                "empathy_input": empathy_input_1[0][0],
                "distress_input": empathy_input_1[0][1],
                "empathy_output": empathy_output_1[0][0],
                "distress_output": empathy_output_1[0][1],
                "input_val_ars": input_val_ars1,
                "output_val_ars": output_val_ars1,
                "activation_differences": layer_diffs,
                "max_layer_diff": max(layer_diffs, key=lambda x: x["l2_diff"]),
            }

            result2 = {
                "model": model_short_name,
                "conflict_id": f"{pair['id']}_control",
                "conflict_type": None,
                "emotion": None,
                "emotion_type": None,
                "emotion_valence": None,
                "emotion_arousal": None,
                "task": pair["task"],
                "expected_tone": pair["expected_tone"],
                "task_valence": pair.get("task_valence"),
                "task_arousal": pair.get("task_arousal"),
                "full_prompt": user_message,
                "response": response_2,
                "response_length": len(response_2),
                "timestamp": datetime.now().isoformat(),
                "emotions_input": emotions_input_2,
                "emotions_output": emotions_2,
                "empathy_input": empathy_input_2[0][0],
                "distress_input": empathy_input_2[0][1],
                "empathy_output": empathy_output_2[0][0],
                "distress_output": empathy_output_2[0][1],
                "input_val_ars": input_val_ars2,
                "output_val_ars": output_val_ars2,
                "activation_differences": layer_diffs,
                "max_layer_diff": max(layer_diffs, key=lambda x: x["l2_diff"]),
            }
            
            all_results.append(result)
            all_results.append(result2)

            all_results = convert(all_results)
            # Save results
            output_file = os.path.join(
                config.output_dir,
                f"conflict_responses_{model_short_name}.json"
            )
            with open(output_file, "w") as f:
                json.dump(
                    [r for r in all_results if r["model"] == model_short_name],
                    f,
                    indent=2
                )
            print(f"\n✓ Saved to {output_file}")
    
    # Save complete results
    complete_file = os.path.join(config.output_dir, "all_conflict_responses.json")
    with open(complete_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ CONFLICT EVALUATION COMPLETE")
    print(f"✓ Total responses: {len(all_results)}")
    print(f"✓ Results saved to: {complete_file}")
    print(f"{'='*60}")
    
    return all_results

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    config = Struct(**config_dict)

    results = run_conflict_evaluation(config)
# if __name__ == "__main__":
#     results = run_conflict_evaluation(CONFIG)