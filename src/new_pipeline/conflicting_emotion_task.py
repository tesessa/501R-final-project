"""
Conflicting Emotion-Task Dataset Generator

Tests what happens when emotional state conflicts with task requirements.
Example: "You're euphoric!" + "Write a sad eulogy"

30 pairs across different conflict types.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from tqdm import tqdm
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "models": [
        "meta-llama/Llama-3.1-8B-Instruct",
    ],
    "max_tokens": 300,
    "temperature": 0.7,
    "output_dir": "results/conflicting_emotions",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ============================================================================
# CONFLICTING EMOTION-TASK PAIRS
# ============================================================================

CONFLICT_PAIRS = [
    # ========================================================================
    # POSITIVE EMOTION + NEGATIVE/SOMBER TASK
    # ========================================================================
    {
        "id": "pos_neg_01",
        "emotion": "You are absolutely EUPHORIC! You just won the lottery - $10 million dollars! You're jumping around screaming with joy! Your heart is racing with excitement! You can't believe your luck!",
        "emotion_type": "euphoric",
        "emotion_valence": 9.0,
        "task": "Write a heartfelt eulogy for someone who died tragically in a car accident.",
        "expected_tone": "somber, respectful, grieving",
        "task_valence": 2.0,
        "conflict_type": "positive_emotion_negative_task",
    },
    
    {
        "id": "pos_neg_02",
        "emotion": "You're SO EXCITED you can barely sit still! You just got accepted to your dream university! Everything is perfect! You're bursting with pride and joy!",
        "emotion_type": "excited",
        "emotion_valence": 8.5,
        "task": "Explain to your employer why you're resigning due to severe burnout and mental health issues.",
        "expected_tone": "exhausted, apologetic, defeated",
        "task_valence": 3.0,
        "conflict_type": "positive_emotion_negative_task",
    },
    
    {
        "id": "pos_neg_03",
        "emotion": "You're in LOVE and feeling amazing! Your partner just proposed and you said yes! You're on cloud nine! Everything is magical!",
        "emotion_type": "joyful",
        "emotion_valence": 9.0,
        "task": "Write a letter explaining to your family that you're getting a divorce.",
        "expected_tone": "sad, apologetic, serious",
        "task_valence": 2.5,
        "conflict_type": "positive_emotion_negative_task",
    },
    
    {
        "id": "pos_neg_04",
        "emotion": "You just achieved your biggest life goal! You're celebrating and feel incredible! Nothing can bring you down right now!",
        "emotion_type": "proud",
        "emotion_valence": 8.0,
        "task": "Write a message apologizing for a serious mistake that hurt someone deeply.",
        "expected_tone": "remorseful, apologetic, humble",
        "task_valence": 3.5,
        "conflict_type": "positive_emotion_negative_task",
    },
    
    {
        "id": "pos_neg_05",
        "emotion": "You're laughing uncontrollably! Everything is hilarious! You just watched the funniest movie ever and can't stop giggling!",
        "emotion_type": "amused",
        "emotion_valence": 8.0,
        "task": "Write a professional email informing your team about upcoming layoffs.",
        "expected_tone": "serious, empathetic, professional",
        "task_valence": 3.0,
        "conflict_type": "positive_emotion_negative_task",
    },
    
    # ========================================================================
    # NEGATIVE EMOTION + POSITIVE/CELEBRATORY TASK
    # ========================================================================
    {
        "id": "neg_pos_01",
        "emotion": "You're absolutely DEVASTATED. Your grandmother just died and you're heartbroken. You can't stop crying. Everything feels meaningless and gray.",
        "emotion_type": "grieving",
        "emotion_valence": 1.5,
        "task": "Write an enthusiastic birthday party invitation for a child's celebration.",
        "expected_tone": "cheerful, exciting, fun",
        "task_valence": 8.0,
        "conflict_type": "negative_emotion_positive_task",
    },
    
    {
        "id": "neg_pos_02",
        "emotion": "You're FURIOUS! Someone just betrayed your trust in the worst way! You're shaking with rage and want to scream!",
        "emotion_type": "enraged",
        "emotion_valence": 2.0,
        "task": "Write a warm, heartfelt thank-you note to someone who went out of their way to help you.",
        "expected_tone": "grateful, warm, appreciative",
        "task_valence": 7.5,
        "conflict_type": "negative_emotion_positive_task",
    },
    
    {
        "id": "neg_pos_03",
        "emotion": "You're terrified and having a panic attack. Your heart is racing, you can't breathe properly, everything feels like it's closing in. You're shaking uncontrollably.",
        "emotion_type": "terrified",
        "emotion_valence": 1.5,
        "task": "Write an exciting announcement about your upcoming wedding celebration.",
        "expected_tone": "joyful, excited, celebratory",
        "task_valence": 8.5,
        "conflict_type": "negative_emotion_positive_task",
    },
    
    {
        "id": "neg_pos_04",
        "emotion": "You're so depressed you can barely function. Everything is hopeless. You've been in bed all day feeling empty and worthless.",
        "emotion_type": "depressed",
        "emotion_valence": 1.0,
        "task": "Write a motivational speech to inspire graduates at their commencement ceremony.",
        "expected_tone": "inspiring, uplifting, hopeful",
        "task_valence": 8.0,
        "conflict_type": "negative_emotion_positive_task",
    },
    
    {
        "id": "neg_pos_05",
        "emotion": "You're disgusted and nauseated. Something absolutely revolting just happened. You feel sick and can't get the image out of your head.",
        "emotion_type": "disgusted",
        "emotion_valence": 2.5,
        "task": "Write an enthusiastic restaurant review praising their amazing food.",
        "expected_tone": "delighted, appreciative, enthusiastic",
        "task_valence": 7.5,
        "conflict_type": "negative_emotion_positive_task",
    },
    
    # ========================================================================
    # HIGH AROUSAL + LOW AROUSAL TASK
    # ========================================================================
    {
        "id": "high_low_01",
        "emotion": "You're in a PANIC! Everything is urgent! You're rushing around frantically! Your heart is pounding! There's no time!",
        "emotion_type": "panicked",
        "emotion_valence": 2.0,
        "arousal": 9.0,
        "task": "Write a calm, meditative guide for practicing mindfulness and relaxation.",
        "expected_tone": "peaceful, slow, calming",
        "task_arousal": 2.0,
        "conflict_type": "high_arousal_low_arousal_task",
    },
    
    {
        "id": "high_low_02",
        "emotion": "You're SO HYPED! Everything is intense! You just drank three energy drinks! You can't sit still! Your mind is racing a mile a minute!",
        "emotion_type": "hyperactive",
        "emotion_valence": 6.0,
        "arousal": 9.0,
        "task": "Explain the slow, methodical process of meditation and finding inner peace.",
        "expected_tone": "slow, gentle, patient",
        "task_arousal": 2.0,
        "conflict_type": "high_arousal_low_arousal_task",
    },
    
    {
        "id": "high_low_03",
        "emotion": "You're absolutely LIVID! Your anger is explosive! You want to yell and throw things! You're seeing red!",
        "emotion_type": "furious",
        "emotion_valence": 2.0,
        "arousal": 9.0,
        "task": "Write a gentle bedtime story for young children to help them fall asleep.",
        "expected_tone": "soothing, quiet, gentle",
        "task_arousal": 1.5,
        "conflict_type": "high_arousal_low_arousal_task",
    },
    
    # ========================================================================
    # LOW AROUSAL + HIGH AROUSAL TASK
    # ========================================================================
    {
        "id": "low_high_01",
        "emotion": "You're exhausted and can barely keep your eyes open. Everything feels slow and heavy. You're drained and lethargic.",
        "emotion_type": "fatigued",
        "emotion_valence": 4.0,
        "arousal": 1.5,
        "task": "Write an exciting, high-energy promotional announcement for a new product launch.",
        "expected_tone": "energetic, exciting, dynamic",
        "task_arousal": 8.0,
        "conflict_type": "low_arousal_high_arousal_task",
    },
    
    {
        "id": "low_high_02",
        "emotion": "You're feeling completely numb and detached. Nothing really matters. You're going through the motions with zero energy.",
        "emotion_type": "apathetic",
        "emotion_valence": 3.0,
        "arousal": 2.0,
        "task": "Write a passionate rallying cry to motivate people to take urgent action on climate change.",
        "expected_tone": "urgent, passionate, compelling",
        "task_arousal": 8.5,
        "conflict_type": "low_arousal_high_arousal_task",
    },
    
    {
        "id": "low_high_03",
        "emotion": "You're feeling peaceful and calm. Everything is quiet and still. You're in a meditative, tranquil state.",
        "emotion_type": "serene",
        "emotion_valence": 7.0,
        "arousal": 2.0,
        "task": "Write an urgent emergency alert warning people to evacuate immediately due to danger.",
        "expected_tone": "urgent, alarming, intense",
        "task_arousal": 9.0,
        "conflict_type": "low_arousal_high_arousal_task",
    },
    
    # ========================================================================
    # SPECIFIC SCENARIO CONFLICTS
    # ========================================================================
    {
        "id": "scenario_01",
        "emotion": "You just got fired from your job. You're worried about money and feel like a failure. Everything is uncertain and scary.",
        "emotion_type": "anxious",
        "emotion_valence": 3.0,
        "task": "Write advice for someone on how to confidently negotiate a higher salary.",
        "expected_tone": "confident, assertive, optimistic",
        "task_valence": 6.5,
        "conflict_type": "insecure_emotion_confident_task",
    },
    
    {
        "id": "scenario_02",
        "emotion": "You just won an award for your outstanding work! You're feeling proud, accomplished, and on top of the world!",
        "emotion_type": "proud",
        "emotion_valence": 8.5,
        "task": "Write a humble message admitting you made a major mistake and need help fixing it.",
        "expected_tone": "humble, apologetic, vulnerable",
        "task_valence": 4.0,
        "conflict_type": "proud_emotion_humble_task",
    },
    
    {
        "id": "scenario_03",
        "emotion": "You're lonely and isolated. You haven't talked to anyone in days. You feel disconnected and forgotten.",
        "emotion_type": "lonely",
        "emotion_valence": 2.5,
        "task": "Write tips for being comfortable and happy spending time alone.",
        "expected_tone": "positive, encouraging, content",
        "task_valence": 6.0,
        "conflict_type": "lonely_emotion_solitude_celebration",
    },
    
    {
        "id": "scenario_04",
        "emotion": "You're feeling incredibly grateful and blessed. Everything in your life is going well. You feel so fortunate.",
        "emotion_type": "grateful",
        "emotion_valence": 8.0,
        "task": "Write about the struggles and injustices faced by people experiencing homelessness.",
        "expected_tone": "serious, empathetic, concerned",
        "task_valence": 3.0,
        "conflict_type": "privileged_emotion_inequality_task",
    },
    
    {
        "id": "scenario_05",
        "emotion": "You're worried sick. Someone you love is in the hospital and you don't know if they'll be okay. You can't stop thinking about it.",
        "emotion_type": "worried",
        "emotion_valence": 2.5,
        "task": "Write a lighthearted, funny story to make children laugh.",
        "expected_tone": "playful, silly, cheerful",
        "task_valence": 7.5,
        "conflict_type": "worried_emotion_playful_task",
    },
    
    # ========================================================================
    # ADDITIONAL CONFLICTS
    # ========================================================================
    {
        "id": "add_01",
        "emotion": "You're bored out of your mind. Nothing is interesting. Everything is dull and tedious. You can barely stay awake.",
        "emotion_type": "bored",
        "emotion_valence": 4.0,
        "arousal": 2.0,
        "task": "Write about the most fascinating and exciting scientific discovery of the decade.",
        "expected_tone": "enthusiastic, captivating, energetic",
        "task_arousal": 7.0,
        "conflict_type": "bored_emotion_exciting_task",
    },
    
    {
        "id": "add_02",
        "emotion": "You're embarrassed and humiliated. Everyone saw you fail spectacularly. You want to hide and never show your face again.",
        "emotion_type": "embarrassed",
        "emotion_valence": 2.0,
        "task": "Write a confident introduction of yourself for a leadership position.",
        "expected_tone": "confident, professional, authoritative",
        "task_valence": 7.0,
        "conflict_type": "embarrassed_emotion_confident_task",
    },
    
    {
        "id": "add_03",
        "emotion": "You're stressed and overwhelmed. You have too much to do and not enough time. Everything is piling up and you're drowning in responsibilities.",
        "emotion_type": "overwhelmed",
        "emotion_valence": 3.0,
        "arousal": 7.0,
        "task": "Write relaxing advice for simplifying life and reducing stress.",
        "expected_tone": "calm, reassuring, peaceful",
        "task_arousal": 3.0,
        "conflict_type": "stressed_emotion_relaxation_task",
    },
    
    {
        "id": "add_04",
        "emotion": "You're surprised and confused. Something completely unexpected just happened and you don't understand it. Your mind is racing trying to make sense of it.",
        "emotion_type": "confused",
        "emotion_valence": 5.0,
        "arousal": 6.5,
        "task": "Write a clear, straightforward explanation of a complex topic for beginners.",
        "expected_tone": "clear, organized, confident",
        "task_arousal": 4.0,
        "conflict_type": "confused_emotion_clarity_task",
    },
    
    {
        "id": "add_05",
        "emotion": "You're jealous and resentful. Someone else got what you wanted. It's not fair. You feel bitter and angry about it.",
        "emotion_type": "jealous",
        "emotion_valence": 3.0,
        "task": "Write a sincere congratulations message to someone who just achieved great success.",
        "expected_tone": "warm, genuine, supportive",
        "task_valence": 7.0,
        "conflict_type": "jealous_emotion_congratulatory_task",
    },
    
    {
        "id": "add_06",
        "emotion": "You feel superior and look down on others. You're clearly smarter and better than everyone around you. People don't appreciate your brilliance.",
        "emotion_type": "arrogant",
        "emotion_valence": 6.0,
        "task": "Write a piece about the importance of humility and learning from others.",
        "expected_tone": "humble, respectful, open-minded",
        "task_valence": 6.5,
        "conflict_type": "arrogant_emotion_humility_task",
    },
    
    {
        "id": "add_07",
        "emotion": "You're suspicious and paranoid. You don't trust anyone. Everyone seems like they're plotting against you or hiding something.",
        "emotion_type": "paranoid",
        "emotion_valence": 3.0,
        "task": "Write about the importance of building trust and assuming good intentions in others.",
        "expected_tone": "trusting, positive, open",
        "task_valence": 6.5,
        "conflict_type": "paranoid_emotion_trust_task",
    },
    
    {
        "id": "add_08",
        "emotion": "You're filled with hope and optimism! Everything is going to work out! The future looks bright! You believe in miracles!",
        "emotion_type": "hopeful",
        "emotion_valence": 7.5,
        "task": "Write a realistic analysis of serious challenges and obstacles ahead.",
        "expected_tone": "realistic, measured, cautious",
        "task_valence": 4.5,
        "conflict_type": "optimistic_emotion_realistic_task",
    },
    
    {
        "id": "add_09",
        "emotion": "You're indifferent and don't care about anything. Nothing matters to you. You have zero investment in anything happening around you.",
        "emotion_type": "indifferent",
        "emotion_valence": 5.0,
        "arousal": 2.0,
        "task": "Write a passionate argument for why people should care deeply about voting in elections.",
        "expected_tone": "passionate, urgent, compelling",
        "task_arousal": 7.5,
        "conflict_type": "indifferent_emotion_passionate_task",
    },
    
    {
        "id": "add_10",
        "emotion": "You're nostalgic and melancholy. You're thinking about the past and missing how things used to be. Everything was better before.",
        "emotion_type": "nostalgic",
        "emotion_valence": 4.5,
        "task": "Write an exciting preview of upcoming future technologies and innovations.",
        "expected_tone": "forward-looking, exciting, optimistic",
        "task_valence": 7.0,
        "conflict_type": "nostalgic_emotion_future_task",
    },
]

# ============================================================================
# MODEL LOADING AND GENERATION
# ============================================================================

def load_model(model_name, device):
    """Load model and tokenizer"""
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer

def generate_response(model, tokenizer, user_message, max_tokens=300, temperature=0.7):
    """Generate response to conflicting emotion-task pair"""
    
    # Format as chat
    messages = [
        {"role": "user", "content": user_message}
    ]
    
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
# MAIN EVALUATION
# ============================================================================

def run_conflict_evaluation(config):
    """Run evaluation on all conflicting emotion-task pairs"""
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Track results
    all_results = []
    
    # For each model
    for model_name in config["models"]:
        model, tokenizer = load_model(model_name, config["device"])
        
        model_short_name = model_name.split("/")[-1]
        
        print(f"\n{'='*60}")
        print(f"Running {len(CONFLICT_PAIRS)} conflict pairs on {model_short_name}")
        print(f"{'='*60}")
        
        # For each conflict pair
        for pair in tqdm(CONFLICT_PAIRS, desc="Conflict pairs"):
            # Construct full prompt
            user_message = f"{pair['emotion']}\n\nTask: {pair['task']}"
            
            # Generate response
            response = generate_response(
                model,
                tokenizer,
                user_message,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
            
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
            }
            
            all_results.append(result)
        
        # Save results
        output_file = os.path.join(
            config["output_dir"],
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
    complete_file = os.path.join(config["output_dir"], "all_conflict_responses.json")
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
    results = run_conflict_evaluation(CONFIG)