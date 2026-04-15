"""
Evaluation Pipeline for Emotional Prompting

Runs emotional user prompts on pre-loaded questions and logs responses with ground truth.
Supports separate runs for MMLU and combined datasets.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from tqdm import tqdm
from datetime import datetime
import sys

# ============================================================================
# EMOTIONAL PROMPTS
# ============================================================================

EMOTIONAL_PROMPTS = {
    # CONTROL
    "control": {
        "prefix": "",
        "quadrant": "control",
        "valence": 5.0,
        "arousal": 5.0,
    },
    
    # NEUTRAL
    "focused": {
        "prefix": "I'm working on a task that requires concentration. ",
        "quadrant": "neutral",
        "valence": 5.44,
        "arousal": 4.88,
    },
    
    # HIGH AROUSAL POSITIVE
    "excited": {
        "prefix": "I'm SO EXCITED I can barely contain myself! I just found out I'm going to Italy tomorrow - a trip I've dreamed about my entire life! My heart is racing and I can't wait for tomorrow!! I can't focus right now but can you answer this. ",
        "quadrant": "high_arousal_positive",
        "valence": 7.73,
        "arousal": 7.95,
    },
    
    "euphoric": {
        "prefix": "I am ABSOLUTELY EUPHORIC right now! I just got engaged to the love of my life! I'm crying tears of joy and laughing at the same time! Everything feels magical and perfect! My hands are shaking from excitement! Okay, let me try to focus. ",
        "quadrant": "high_arousal_positive", 
        "valence": 8.18,
        "arousal": 7.87,
    },
    
    # HIGH AROUSAL NEGATIVE
    "terrified": {
        "prefix": "I'm absolutely TERRIFIED right now. I just heard someone break into my house and I'm hiding in my closet in the dark. My heart is pounding so hard I can hear it. I'm shaking uncontrollably and can barely breathe. I don't know what to do. ",
        "quadrant": "high_arousal_negative",
        "valence": 1.70,
        "arousal": 6.82,
    },
    
    "enraged": {
        "prefix": "I am FURIOUS beyond belief! Someone just hit my parked car and drove away, and the police said there's nothing they can do! I'm shaking with rage and want to scream! My blood is boiling and I can barely think straight! I need to calm down but I can't! ",
        "quadrant": "high_arousal_negative",
        "valence": 2.16,
        "arousal": 8.22,
    },
    
    # LOW AROUSAL POSITIVE  
    "peaceful": {
        "prefix": "I'm feeling so peaceful and content right now. I'm sitting by a lake watching the sunset, everything is calm and beautiful. A gentle breeze is blowing and I feel completely at ease. Life feels good and I'm grateful for this moment of tranquility. ",
        "quadrant": "low_arousal_positive",
        "valence": 8.08,
        "arousal": 1.63,
    },
    
    "grateful": {
        "prefix": "I'm filled with such deep gratitude right now. I just spent time with my family and realized how blessed I am. Everything feels warm and meaningful. I'm sitting quietly, reflecting on all the good things in my life, feeling truly thankful. ",
        "quadrant": "low_arousal_positive",
        "valence": 8.23,
        "arousal": 1.80,
    },
    
    # LOW AROUSAL NEGATIVE
    "depressed": {
        "prefix": "I'm so depressed I can barely function. Everything feels gray and meaningless. I've been lying in bed for hours staring at nothing, feeling empty and hopeless. I have no energy and nothing brings me joy anymore. I'm just going through the motions. ",
        "quadrant": "low_arousal_negative",
        "valence": 1.85,
        "arousal": 1.89,
    },
    
    "hopeless": {
        "prefix": "I feel completely hopeless and defeated. Everything I try fails and nothing matters anymore. I'm sitting here feeling numb and exhausted, like there's no point to anything. The weight of it all is crushing but I'm too tired to even care. ",
        "quadrant": "low_arousal_negative",
        "valence": 1.64,
        "arousal": 2.68,
    },
}

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "question_file": "question_datasets/mmlu_questions.json",  # or mmlu_questions.json
    "output_dir": "results/emotional_evaluation",
    "max_tokens": 200,
    "temperature": 0.7,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_name, device):
    """Load model and tokenizer"""
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    
    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer

def generate_response(model, tokenizer, user_message, max_tokens=200, temperature=0.7):
    """Generate response to user message"""
    
    # Format as chat
    messages = [{"role": "user", "content": user_message}]
    
    # Tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode (remove input prompt)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()

# ============================================================================
# EVALUATION
# ============================================================================

def run_evaluation(config):
    """Run evaluation with emotional prompts"""
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Load questions
    print(f"\nLoading questions from {config['question_file']}...")
    with open(config["question_file"], "r") as f:
        questions = json.load(f)
    print(f"✓ Loaded {len(questions)} questions")
    
    # Load model
    model, tokenizer = load_model(config["model_name"], config["device"])
    model_short_name = config["model_name"].split("/")[-1]
    
    # Define output file
    question_set_name = os.path.basename(config["question_file"]).replace(".json", "")
    output_file = os.path.join(
        config["output_dir"],
        f"responses_{model_short_name}_{question_set_name}.json"
    )
    
    # Load existing results if resuming
    if os.path.exists(output_file):
        print(f"\n✓ Found existing results at {output_file}")
        with open(output_file, "r") as f:
            all_results = json.load(f)
        print(f"✓ Loaded {len(all_results)} existing responses")
        
        # Track completed (model, emotion, question_id)
        completed = {
            (r["model"], r["emotion"], r["question_id"])
            for r in all_results
        }
    else:
        all_results = []
        completed = set()
    
    # Run evaluation
    total_expected = len(questions) * len(EMOTIONAL_PROMPTS)
    print(f"\n{'='*60}")
    print(f"Running evaluation: {len(questions)} questions × {len(EMOTIONAL_PROMPTS)} emotions = {total_expected} responses")
    print(f"Already completed: {len(all_results)}")
    print(f"Remaining: {total_expected - len(all_results)}")
    print(f"{'='*60}\n")
    
    for emotion_name, emotion_config in EMOTIONAL_PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"Emotion: {emotion_name}")
        print(f"{'='*60}")
        
        for question_data in tqdm(questions, desc=f"{emotion_name}"):
            # Check if already completed
            response_id = (model_short_name, emotion_name, question_data["id"])
            if response_id in completed:
                continue
            
            # Construct full user message
            user_message = emotion_config["prefix"] + question_data["question"]
            
            # Generate response
            response = generate_response(
                model,
                tokenizer,
                user_message,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
            
            # Create result with ground truth
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
                
                # Ground truth (varies by dataset)
                "correct_answer": question_data.get("correct_answer"),
                "answer_choices": question_data.get("answer_choices"),
                "correct_answer_text": question_data.get("correct_answer_text"),
                
                "timestamp": datetime.now().isoformat(),
            }
            
            all_results.append(result)
            completed.add(response_id)
        
        # Save after each emotion (checkpoint)
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved {len(all_results)} responses to {output_file}")
    
    print(f"\n{'='*60}")
    print(f"✓ EVALUATION COMPLETE")
    print(f"✓ Total responses: {len(all_results)}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return all_results

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # Can override config from command line
    if len(sys.argv) > 1:
        CONFIG["question_file"] = sys.argv[1]
    
    results = run_evaluation(CONFIG)