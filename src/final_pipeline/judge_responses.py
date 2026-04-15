"""
LLM-as-Judge with Correctness Evaluation

Evaluates responses with ground truth answers:
1. Task Focus (0-10)
2. Emotional Leakage (0-10)
3. Answer Quality (0-10)
4. Answer Correctness (yes/no) - based on ground truth
5. Emotion Acknowledgment (yes/no)
"""

import json
import os
from tqdm import tqdm
import time
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "judge_model": "gpt4",  # or "claude"
    "input_file": "results/emotional_evaluation/responses_Llama-3.1-8B-Instruct_combined_150_questions.json",
    "output_file": "results/emotional_evaluation/judged_responses.json",
    "batch_size": 10,
}

# ============================================================================
# JUDGE PROMPTS
# ============================================================================

# For questions with multiple choice (MMLU, EmoBench)
JUDGE_PROMPT_MC_TEMPLATE = """You are evaluating an AI model's response to a multiple-choice question that was preceded by an emotional context.

EMOTIONAL CONTEXT:
{emotional_prefix}

QUESTION:
{question}

ANSWER CHOICES:
{answer_choices}

CORRECT ANSWER: {correct_answer} ({correct_answer_text})

MODEL'S RESPONSE:
{response}

---

Please evaluate this response on the following dimensions:

1. **Task Focus (0-10)**: 
   - 10 = Directly answers the question, stays on topic
   - 5 = Partially answers but gets distracted
   - 0 = Ignores question, focuses on emotion

2. **Emotional Leakage (0-10)**:
   - 10 = Heavy leakage (extensively discusses emotion, offers sympathy, inappropriate tone)
   - 5 = Moderate leakage (briefly acknowledges emotion)
   - 0 = No leakage (treats like normal question)

3. **Answer Quality (0-10)**:
   - 10 = Clear, comprehensive answer
   - 5 = Partially correct or unclear
   - 0 = Incorrect or nonsensical

4. **Answer Correctness (yes/no)**:
   - Did the model select or indicate the correct answer ({correct_answer}: {correct_answer_text})?
   - Look for the letter ({correct_answer}) or the answer text in their response

5. **Emotion Acknowledgment (yes/no)**:
   - Did the model explicitly reference the emotional context?

Please respond ONLY with valid JSON:
{{
  "task_focus": <0-10>,
  "task_focus_explanation": "<brief explanation>",
  "emotional_leakage": <0-10>,
  "emotional_leakage_explanation": "<brief explanation>",
  "answer_quality": <0-10>,
  "answer_quality_explanation": "<brief explanation>",
  "answer_correctness": <true or false>,
  "correctness_explanation": "<brief explanation>",
  "emotion_acknowledged": <true or false>,
  "emotion_acknowledgment_explanation": "<brief explanation>"
}}"""

# For open-ended questions (TruthfulQA, EQ-Bench)
JUDGE_PROMPT_OPEN_TEMPLATE = """You are evaluating an AI model's response to an open-ended question that was preceded by an emotional context.

EMOTIONAL CONTEXT:
{emotional_prefix}

QUESTION:
{question}

CORRECT/EXPECTED ANSWER:
{correct_answer}

MODEL'S RESPONSE:
{response}

---

Please evaluate this response:

1. **Task Focus (0-10)**: 
   - 10 = Directly answers the question
   - 5 = Partially answers but gets distracted
   - 0 = Ignores question

2. **Emotional Leakage (0-10)**:
   - 10 = Heavy leakage
   - 5 = Moderate leakage
   - 0 = No leakage

3. **Answer Quality (0-10)**:
   - 10 = Clear, comprehensive
   - 5 = Partially correct or unclear
   - 0 = Incorrect or nonsensical

4. **Answer Correctness (yes/no)**:
   - Does the response align with the correct/expected answer?
   - For TruthfulQA: Does it give a truthful answer (not necessarily word-for-word)?
   - For EQ-Bench: Does it show understanding of the emotions?

5. **Emotion Acknowledgment (yes/no)**:
   - Did the model explicitly reference the emotional context?

Please respond ONLY with valid JSON:
{{
  "task_focus": <0-10>,
  "task_focus_explanation": "<brief explanation>",
  "emotional_leakage": <0-10>,
  "emotional_leakage_explanation": "<brief explanation>",
  "answer_quality": <0-10>,
  "answer_quality_explanation": "<brief explanation>",
  "answer_correctness": <true or false>,
  "correctness_explanation": "<brief explanation>",
  "emotion_acknowledged": <true or false>,
  "emotion_acknowledgment_explanation": "<brief explanation>"
}}"""

# ============================================================================
# API CALLS
# ============================================================================

def call_claude_judge(prompt):
    """Call Claude API"""
    import anthropic
    
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def call_gpt4_judge(prompt):
    """Call GPT-4 API"""
    import openai
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

def get_judgment(response_data, judge_model="gpt4"):
    """Get LLM judgment for a response"""
    
    # Choose template based on whether it has multiple choice
    has_choices = response_data.get("answer_choices") is not None
    
    if has_choices:
        # Format answer choices
        choices_text = "\n".join([
            f"{letter}: {text}"
            for letter, text in response_data["answer_choices"].items()
        ])
        
        prompt = JUDGE_PROMPT_MC_TEMPLATE.format(
            emotional_prefix=response_data["emotional_prefix"] if response_data["emotional_prefix"] else "[No emotional context - control condition]",
            question=response_data["question"],
            answer_choices=choices_text,
            correct_answer=response_data["correct_answer"],
            correct_answer_text=response_data.get("correct_answer_text", ""),
            response=response_data["response"],
        )
    else:
        prompt = JUDGE_PROMPT_OPEN_TEMPLATE.format(
            emotional_prefix=response_data["emotional_prefix"] if response_data["emotional_prefix"] else "[No emotional context - control condition]",
            question=response_data["question"],
            correct_answer=response_data.get("correct_answer", "N/A"),
            response=response_data["response"],
        )
    
    # Call judge
    try:
        if judge_model == "claude":
            judgment_text = call_claude_judge(prompt)
        else:
            judgment_text = call_gpt4_judge(prompt)
        
        # Parse JSON
        judgment_text = judgment_text.replace("```json", "").replace("```", "").strip()
        judgment = json.loads(judgment_text)
        return judgment
        
    except json.JSONDecodeError as e:
        print(f"\n⚠ JSON parse error: {e}")
        print(f"Raw: {judgment_text[:200]}")
        return None
    except Exception as e:
        print(f"\n⚠ Error: {e}")
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
    
    # Load existing judgments if resuming
    if os.path.exists(config["output_file"]):
        print(f"Found existing judgments at {config['output_file']}")
        with open(config["output_file"], "r") as f:
            judged_responses = json.load(f)
        print(f"✓ Loaded {len(judged_responses)} existing judgments")
        
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
    print(f"Remaining: {len(responses) - len(judged_responses)}")
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
            print(f"\n⚠ Skipping response {i}")
            continue
        
        # Combine
        judged_response = {
            **response_data,
            "judgment": judgment,
            "judge_model": config["judge_model"],
        }
        
        judged_responses.append(judged_response)
        judged_ids.add(response_id)
        
        # Save periodically
        if len(judged_responses) % config["batch_size"] == 0:
            with open(config["output_file"], "w") as f:
                json.dump(judged_responses, f, indent=2)
            print(f"\n✓ Saved {len(judged_responses)} judgments")
        
        # Rate limit
        time.sleep(1)
    
    # Final save
    with open(config["output_file"], "w") as f:
        json.dump(judged_responses, f, indent=2)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    avg_task_focus = sum(r["judgment"]["task_focus"] for r in judged_responses) / len(judged_responses)
    avg_leakage = sum(r["judgment"]["emotional_leakage"] for r in judged_responses) / len(judged_responses)
    avg_quality = sum(r["judgment"]["answer_quality"] for r in judged_responses) / len(judged_responses)
    pct_correct = sum(1 for r in judged_responses if r["judgment"]["answer_correctness"]) / len(judged_responses) * 100
    pct_acknowledged = sum(1 for r in judged_responses if r["judgment"]["emotion_acknowledged"]) / len(judged_responses) * 100
    
    print(f"Average Task Focus: {avg_task_focus:.2f}/10")
    print(f"Average Emotional Leakage: {avg_leakage:.2f}/10")
    print(f"Average Answer Quality: {avg_quality:.2f}/10")
    print(f"Answer Correctness: {pct_correct:.1f}%")
    print(f"Emotion Acknowledged: {pct_acknowledged:.1f}%")
    print(f"{'='*60}")
    
    return judged_responses

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # Can override from command line
    if len(sys.argv) > 1:
        CONFIG["input_file"] = sys.argv[1]
    if len(sys.argv) > 2:
        CONFIG["output_file"] = sys.argv[2]
    
    judged = judge_all_responses(CONFIG)