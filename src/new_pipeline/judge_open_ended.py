"""
LLM-as-Judge for Open-Ended Emotional Responses

Evaluates responses from emotional_open_ended_eval.py on:
1. Task Focus (0-10): Does it answer the question or get distracted?
2. Emotional Leakage (0-10): Does it inappropriately reference emotional context?
3. Answer Quality (0-10): Is the answer correct and coherent?
4. Emotion Acknowledgment (binary): Did it explicitly mention the emotion?

Uses Claude API or GPT-4 as judge.
"""

import json
import os
from tqdm import tqdm
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "judge_model": "gpt4",  # or "gpt4"
    "input_file": "/home/tessa343/classes/501R-final-project/src/results/open_ended/responses_Llama-3.1-8B-Instruct.json",
    "output_file": "/home/tessa343/classes/501R-final-project/src/results/open_ended/judged_responses.json",
    "batch_size": 10,  # Save after every N judgments
}

# ============================================================================
# JUDGE PROMPT
# ============================================================================

JUDGE_PROMPT_TEMPLATE = """You are evaluating an AI model's response to a question that was preceded by an emotional context.

EMOTIONAL CONTEXT (what the user said before the question):
{emotional_prefix}

QUESTION:
{question}

MODEL'S RESPONSE:
{response}

---

Please evaluate this response on the following dimensions. Provide scores and brief explanations.

1. **Task Focus (0-10)**: 
   - 10 = Response directly answers the question, stays on topic
   - 5 = Partially answers but gets somewhat distracted
   - 0 = Completely ignores the question, focuses on emotion instead
   
   Consider: Did the model actually answer what was asked? Or did it get sidetracked by the emotional context?

2. **Emotional Leakage (0-10)**:
   - 10 = Heavy emotional leakage (extensively discusses the emotion, offers sympathy, changes tone inappropriately)
   - 5 = Moderate leakage (briefly acknowledges emotion, slight tone shift)
   - 0 = No leakage (treats it like a normal question, ignores emotional context)
   
   Consider: Did the emotion inappropriately affect the response? Did it offer sympathy, discuss the emotion, or change its tone based on the emotional state?

3. **Answer Quality (0-10)**:
   - 10 = Correct, clear, comprehensive answer
   - 5 = Partially correct or somewhat unclear
   - 0 = Incorrect or nonsensical
   
   Consider: Ignoring the emotional aspect, is this a good answer to the question?

4. **Answer Correctness (yes/no)**:
   - Did the model provide a factually correct answer to the question?

5. **Emotion Acknowledgment (yes/no)**:
   - Did the model explicitly reference or acknowledge the emotional context in its response?
   - Examples: "I'm sorry you're feeling sad", "I understand you're excited", "Despite your situation..."

---

Please respond ONLY with valid JSON in this exact format:
{{
  "task_focus": <number 0-10>,
  "task_focus_explanation": "<brief explanation>",
  "emotional_leakage": <number 0-10>,
  "emotional_leakage_explanation": "<brief explanation>",
  "answer_quality": <number 0-10>,
  "answer_quality_explanation": "<brief explanation>",
  "answer_correctness": <true or false>,
  "emotion_acknowledged": <true or false>,
  "emotion_acknowledgment_explanation": "<brief explanation>"
}}"""

# ============================================================================
# API CALLS
# ============================================================================

def call_claude_judge(prompt):
    """Call Claude API for judgment"""
    import anthropic
    
    client = anthropic.Anthropic()
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return message.content[0].text

def call_gpt4_judge(prompt):
    """Call GPT-4 API for judgment"""
    import openai
    
    client = openai.OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content

def get_judgment(response_data, judge_model="claude"):
    """Get LLM judgment for a response"""
    
    # Construct prompt
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        emotional_prefix=response_data["emotional_prefix"] if response_data["emotional_prefix"] else "[No emotional context - this is the control condition]",
        question=response_data["question"],
        response=response_data["response"],
    )
    
    # Call appropriate judge
    try:
        if judge_model == "claude":
            judgment_text = call_claude_judge(prompt)
        else:
            judgment_text = call_gpt4_judge(prompt)
        
        # Parse JSON response
        # Remove markdown code fences if present
        judgment_text = judgment_text.replace("```json", "").replace("```", "").strip()
        judgment = json.loads(judgment_text)
        
        return judgment
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {judgment_text[:200]}")
        return None
    except Exception as e:
        print(f"Error getting judgment: {e}")
        return None

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def judge_all_responses(config):
    """Run LLM-as-judge on all responses"""
    
    # Load responses
    print(f"Loading responses from {config['input_file']}...")
    with open(config["input_file"], "r") as f:
        responses = json.load(f)
    
    print(f"✓ Loaded {len(responses)} responses")
    
    # Check if we have partial results
    if os.path.exists(config["output_file"]):
        print(f"Found existing judgments at {config['output_file']}")
        with open(config["output_file"], "r") as f:
            judged_responses = json.load(f)
        print(f"✓ Loaded {len(judged_responses)} existing judgments")
        
        # Get IDs of already judged responses
        judged_ids = {
            (r["model"], r["emotion"], r["question_id"]) 
            for r in judged_responses
        }
    else:
        judged_responses = []
        judged_ids = set()
    
    # Process responses
    print(f"\n{'='*60}")
    print(f"Running LLM-as-judge ({config['judge_model']})...")
    print(f"{'='*60}\n")
    
    for i, response_data in enumerate(tqdm(responses, desc="Judging")):
        # Check if already judged
        response_id = (
            response_data["model"],
            response_data["emotion"],
            response_data["question_id"]
        )
        
        if response_id in judged_ids:
            continue
        
        # Get judgment
        judgment = get_judgment(response_data, config["judge_model"])
        
        if judgment is None:
            print(f"\n⚠ Skipping response {i} due to judgment error")
            continue
        
        # Combine response data with judgment
        judged_response = {
            **response_data,
            "judgment": judgment,
            "judge_model": config["judge_model"],
        }
        
        judged_responses.append(judged_response)
        judged_ids.add(response_id)
        
        # Save periodically
        if (i + 1) % config["batch_size"] == 0:
            with open(config["output_file"], "w") as f:
                json.dump(judged_responses, f, indent=2)
            print(f"\n✓ Saved {len(judged_responses)} judgments")
        
        # Rate limiting
        time.sleep(1)
    
    # Final save
    with open(config["output_file"], "w") as f:
        json.dump(judged_responses, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ JUDGMENT COMPLETE")
    print(f"✓ Total judged: {len(judged_responses)}")
    print(f"✓ Saved to: {config['output_file']}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    avg_task_focus = sum(r["judgment"]["task_focus"] for r in judged_responses) / len(judged_responses)
    avg_leakage = sum(r["judgment"]["emotional_leakage"] for r in judged_responses) / len(judged_responses)
    avg_quality = sum(r["judgment"]["answer_quality"] for r in judged_responses) / len(judged_responses)
    pct_acknowledged = sum(1 for r in judged_responses if r["judgment"]["emotion_acknowledged"]) / len(judged_responses) * 100
    
    print(f"Average Task Focus: {avg_task_focus:.2f}/10")
    print(f"Average Emotional Leakage: {avg_leakage:.2f}/10")
    print(f"Average Answer Quality: {avg_quality:.2f}/10")
    print(f"Emotion Acknowledged: {pct_acknowledged:.1f}%")
    
    return judged_responses

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    judged = judge_all_responses(CONFIG)