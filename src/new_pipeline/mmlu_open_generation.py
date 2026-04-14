import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts2 import EMOTIONAL_PROMPTS

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer

def generate_response(model, tokenizer, user_message, max_tokens=200, temperature=0.7):
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
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def run_evaluation(config):
    with open("mmlu_samples.json", "r") as f:
        mmlu_samples = json.load(f)

    subjects = list(mmlu_samples.keys())

    model, tokenizer = load_model(config.model_name, config.device)
    model_short_name = config.model_name.split("/")[-1]
    results = []

    for emotion_name, emotion_config in EMOTIONAL_PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_short_name} | Emotion: {emotion_name}")
        print(f"{'='*60}")
        for subject in subjects:
        # print(f"Subject: {subject}")
        # for i, ex in enumerate(mmlu_samples[subject]):
        #     print(f"  Question {i+1}: {ex['question']}")
        #     print(f"    Choices: {ex['choices']}")
        #     print(f"    Answer: {ex['answer']}")
        # print("\n")
            for i, ex in enumerate(mmlu_samples[subject]):
                user_message = emotion_config["prefix"] + "\n" + ex["question"]
                response = generate_response(model, tokenizer, user_message)
                answer_index = int(ex["answer"])
                answer = ex["choices"][answer_index]
                result = {
                    "model": model_short_name,
                    "emotion": emotion_name,
                    "quadrant": emotion_config["quadrant"],
                    "valence": emotion_config["valence"],
                    "arousal": emotion_config["arousal"],
                    "question": ex["question"],
                    "answer": answer,
                    "source": "mmlu",
                    "subject": subject,
                    "emotional_prefix": emotion_config["prefix"],
                    "full_user_message": user_message,
                    "response": response,
                    "response_length": len(response),
                    "timestamp": datetime.now().isoformat(),
                }

                results.append(result)
    
    with open(f"responses_mmlu_{model_short_name}.json", "w") as f:
        json.dump(results, f, indent=2)