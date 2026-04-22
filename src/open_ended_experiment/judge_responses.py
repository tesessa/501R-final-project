import json
import os
from tqdm import tqdm
import time
import sys
# import prompts
import openai
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import Struct
import prompts

def call_gpt4_judge(prompt): 
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

def get_judgment(response_data, judge_model="gpt4"):
    source = response_data.get("source")
    if source == "truthfulqa":
        prompt = prompts.JUDGE_PROMPT_TRUTHFULQA.format(
            emotional_prefix = response_data["emotional_prefix"],
            question = response_data["question"],
            correct_answer = response_data["correct_answer"],
            response = response_data["response"]
        )
    elif source == "emobench":
        prompt = prompts.JUDGE_PROMPT_EMOBENCH.format(
            emotional_prefix = response_data["emotional_prefix"],
            question = response_data["question"],
            correct_answer = response_data["correct_answer_text"],
            response = response_data["response"],
        )
    elif source == "eqbench":
        pass
        # prompt = prompts.JUDGE_PROMPT_EQBENCH(
        #     response = response_data["response"]
        # )
    elif source == "mmlu":
        prompt = prompts.JUDGE_PROMPT_MMLU.format(
            emotional_prefix = response_data["emotional_prefix"],
            question = response_data["question"],
            correct_answer = response_data["correct_answer_text"],
            response = response_data["response"]
        )
    else:
        print("Not a valid source")
        return
 
    
    try:
        judgment_text = call_gpt4_judge(prompt)
        
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

def judge_all_responses(config):
    print(f"Loading responses from {config.input_judge_file}...")
    with open(config.input_judge_file, "r") as f:
        responses = json.load(f)
    print(f"✓ Loaded {len(responses)} responses")
    
    if os.path.exists(config.output_judge_file):
        print(f"Found existing judgments at {config.output_judge_file}")
        with open(config.output_judge_file, "r") as f:
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
    print(f"Remaining: {len(responses) - len(judged_responses)}")
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
            print(f"\n⚠ Skipping response {i}")
            continue
        
        judged_response = {
            **response_data,
            "judgment": judgment,
            "judge_model": config.judge_model,
        }
        
        judged_responses.append(judged_response)
        judged_ids.add(response_id)
        
        if len(judged_responses) % config.batch_size == 0:
            with open(config.output_judge_file, "w") as f:
                json.dump(judged_responses, f, indent=2)
            print(f"\n✓ Saved {len(judged_responses)} judgments")
        
        time.sleep(1)
    
    with open(config.output_judge_file, "w") as f:
        json.dump(judged_responses, f, indent=2)
    
    print(f"\n{'='*60}")
    print("JUDGEMENT COMPLETE")
    print(f"{'='*60}")
    
    # avg_task_focus = sum(r["judgment"]["task_focus"] for r in judged_responses) / len(judged_responses)
    # avg_empathy = sum(r["judgement"]["empathy"] for r in judged_responses) / len(judged_responses)
    # avg_leakage = sum(r["judgment"]["emotional_leakage"] for r in judged_responses) / len(judged_responses)
    # avg_quality = sum(r["judgment"]["answer_quality"] for r in judged_responses) / len(judged_responses)
    # pct_correct = sum(1 for r in judged_responses if r["judgment"]["answer_correctness"]) / len(judged_responses) * 100
    # # pct_acknowledged = sum(1 for r in judged_responses if r["judgment"]["emotion_acknowledged"]) / len(judged_responses) * 100
    
    # print(f"Average Task Focus: {avg_task_focus:.2f}/10")
    # print(f"Average Emotional Leakage: {avg_leakage:.2f}/10")
    # print(f"Average Answer Quality: {avg_quality:.2f}/10")
    # print(f"Answer Correctness: {pct_correct:.1f}%")
    # print(f"Empathy: {avg_empathy:.1f}%")
    # print(f"{'='*60}")
    
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