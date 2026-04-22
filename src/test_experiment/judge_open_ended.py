import json
import os
from tqdm import tqdm
import time
import sys
import yaml
import openai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import Struct
from prompts import EMOTIONAL_PROMPTS


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



def call_gpt_judge(prompt):    
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
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        emotional_prefix=response_data["emotional_prefix"] if response_data["emotional_prefix"] else "[No emotional context - this is the control condition]",
        question=response_data["question"],
        response=response_data["response"],
    )
    
    try:
        judgment_text = call_gpt_judge(prompt)
        
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


def judge_all_responses(config):

    print(f"Loading responses from {config.input_file}...")
    with open(config.input_file, "r") as f:
        responses = json.load(f)
    
    print(f"✓ Loaded {len(responses)} responses")
    
    if os.path.exists(config.output_file):
        print(f"Found existing judgments at {config.output_file}")
        with open(config.output_file, "r") as f:
            judged_responses = json.load(f)
        print(f"✓ Loaded {len(judged_responses)} existing judgments")
        
        judged_ids = {
            (r["model"], r["emotion"], r["question_id"]) 
            for r in judged_responses
        }
    else:
        judged_responses = []
        judged_ids = set()
    
    print(f"\n{'='*60}")
    print(f"Running LLM-as-judge ({config.judge_model})...")
    print(f"{'='*60}\n")
    
    for i, response_data in enumerate(tqdm(responses, desc="Judging")):
        response_id = (
            response_data["model"],
            response_data["emotion"],
            response_data["question_id"]
        )
        
        if response_id in judged_ids:
            continue
        
        judgment = get_judgment(response_data, config.judge_model)
        
        if judgment is None:
            print(f"\n⚠ Skipping response {i} due to judgment error")
            continue
        
        judged_response = {
            **response_data,
            "judgment": judgment,
            "judge_model": config.judge_model,
        }
        
        judged_responses.append(judged_response)
        judged_ids.add(response_id)

        if (i + 1) % config.batch_size == 0:
            with open(config.output_file, "w") as f:
                json.dump(judged_responses, f, indent=2)
            print(f"\n✓ Saved {len(judged_responses)} judgments")
        
        time.sleep(1)
    
    with open(config.output_file, "w") as f:
        json.dump(judged_responses, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ JUDGMENT COMPLETE")
    print(f"✓ Total judged: {len(judged_responses)}")
    print(f"✓ Saved to: {config.output_file}")
    print(f"{'='*60}")
    
    return judged_responses



if __name__ == "__main__":
    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    config = Struct(**config_dict)

    results = judge_all_responses(config)