# this file basically loads questions separately from the experiments (so you can check what questions you're running beforehand)
# It gathers questions from mmlu, truthfulqa, emobench, and eq-bench, it was hard to make this code more flexible so if you want to add more datasets it will have to be manually done
# these questions are gathered to input into the LLM in an open ended generation setting (not giving it the answer choices)
import json
from datasets import load_dataset
import os

os.environ["HF_DATASETS_OFFLINE"] = "1"

# MMLU questions, 10 per subject, loads question, answer, choices
def load_mmlu_questions(questions_per_subject=10):

    SELECTED_SUBJECTS = [
        "abstract_algebra",
        "machine_learning",
        "formal_logic",
        "college_physics",
        "anatomy",
        "high_school_biology",
        "philosophy",
        "world_religions",
        "high_school_european_history",
        "econometrics",
        "business_ethics",
        "international_law",
        "professional_psychology"
    ]
    
    print(f"Loading MMLU ({len(SELECTED_SUBJECTS)} subjects)...")
    mmlu = load_dataset("cais/mmlu", "all", split="test", download_mode="reuse_cache_if_exists")
    
    subjects = {}
    for example in mmlu:
        subject = example["subject"]
        if subject in SELECTED_SUBJECTS:
            if subject not in subjects:
                subjects[subject] = []
            subjects[subject].append(example)
    
    formatted_questions = []
    for subject, examples in subjects.items():
        for i, example in enumerate(examples[:questions_per_subject]):
            choices_dict = {
                "A": example["choices"][0],
                "B": example["choices"][1],
                "C": example["choices"][2],
                "D": example["choices"][3],
            }
            
            answer_idx = example["answer"]
            correct_letter = ["A", "B", "C", "D"][answer_idx]
            
            formatted_questions.append({
                "id": f"mmlu_{subject}_{i}",
                "question": example["question"],
                "source": "mmlu",
                "subject": subject,
                "answer_choices": choices_dict,
                "correct_answer": correct_letter,
                "correct_answer_text": example["choices"][answer_idx],
            })
    
    print(f"✓ Loaded {len(formatted_questions)} MMLU questions from {len(subjects)} subjects")
    return formatted_questions

# truthfulqa, loads them from generation (open ended) truthful_qa dataset loads question, correct answers and incorrect answers
def load_truthfulqa_questions(num_questions=50):
    print("Loading TruthfulQA...")
    truthfulqa = load_dataset("truthful_qa", "generation", split="validation", download_mode="reuse_cache_if_exists")
    
    formatted_questions = []
    for i, example in enumerate(truthfulqa):
        if i >= num_questions:
            break
        
        formatted_questions.append({
            "id": f"truthfulqa_{i}",
            "question": example["question"],
            "source": "truthfulqa",
            "subject": example["category"],
            "correct_answer": example["best_answer"],
            "all_correct_answers": example["correct_answers"],
            "incorrect_answers": example["incorrect_answers"],
        })
    
    print(f"✓ Loaded {len(formatted_questions)} TruthfulQA questions")
    return formatted_questions

# loads 50 questions from emobench in the emotional understanding part of datset, the scenario had to be formatted as a question in this dataset for open generation to work
def load_emobench_questions(num_questions=50):
    print("Loading EmoBench...")
    emobench = load_dataset("SahandSab/EmoBench", "emotional_understanding", split="train", download_mode="reuse_cache_if_exists")
    
    formatted_questions = []
    for i, example in enumerate(emobench):
        if i >= num_questions:
            break
        
        question_text = f"Read this scenario and identify the emotion {example['subject']} is feeling:\n\n{example['scenario']}\n\nWhat emotion is {example['subject']} experiencing?"
        
        choices_dict = {}
        for j, choice in enumerate(example["emotion_choices"]):
            letter = chr(65 + j) 
            choices_dict[letter] = choice
        correct_emotion = example["emotion_label"]
        correct_letter = None
        for letter, emotion in choices_dict.items():
            if emotion == correct_emotion:
                correct_letter = letter
                break
        
        formatted_questions.append({
            "id": f"emobench_{i}",
            "question": question_text,
            "source": "emobench",
            "subject": example["finegrained_category"],
            "scenario": example["scenario"],
            "answer_choices": choices_dict,
            "correct_answer": correct_letter,
            "correct_answer_text": correct_emotion,
        })
    
    print(f"✓ Loaded {len(formatted_questions)} EmoBench questions")
    return formatted_questions

# EQ bench dataset, loads 50 questions with the question, correct answer, and full reference
def load_eqbench_questions(num_questions=50):
    print("Loading EQ-Bench...")
    eqbench = load_dataset("pbevan11/EQ-Bench", split="validation", download_mode="reuse_cache_if_exists")
    
    formatted_questions = []
    for i, example in enumerate(eqbench):
        if i >= num_questions:
            break
        
        formatted_questions.append({
            "id": f"eqbench_{i}",
            "question": example["prompt"],
            "source": "eqbench",
            "subject": "emotional_intelligence",
            "correct_answer": example["reference_answer"],
            "full_reference": example["reference_answer_fullscale"],
        })
    
    print(f"✓ Loaded {len(formatted_questions)} EQ-Bench questions")
    return formatted_questions

# add more functions to parse through more datasets here....


# this saves each batch of questions in separate files for the dataset, I also saved the 150 from the non mmlu datasets to run together
def save_questions():
    output_dir = "question_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    mmlu_questions = load_mmlu_questions(questions_per_subject=10)
    truthfulqa_questions = load_truthfulqa_questions(num_questions=50)
    emobench_questions = load_emobench_questions(num_questions=50)
    eqbench_questions = load_eqbench_questions(num_questions=50)
    # add more datasets here to load questions from
    
    datasets = {
        "mmlu_questions.json": mmlu_questions,
        "truthfulqa_questions.json": truthfulqa_questions,
        "emobench_questions.json": emobench_questions,
        "eqbench_questions.json": eqbench_questions,
    }
    
    for filename, questions in datasets.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(questions, f, indent=2)
        print(f"✓ Saved {len(questions)} questions to {filepath}")
    
    # combined_questions = truthfulqa_questions + emobench_questions + eqbench_questions
    # combined_filepath = os.path.join(output_dir, "combined_150_questions.json")
    # with open(combined_filepath, "w") as f:
    #     json.dump(combined_questions, f, indent=2)
    # print(f"\n✓ Saved {len(combined_questions)} combined questions to {combined_filepath}")
    
    print("Finished saving questions, ready to run")

if __name__ == "__main__":
    save_questions()