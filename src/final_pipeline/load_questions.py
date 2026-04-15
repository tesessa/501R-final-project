"""
Load and format questions from different datasets into standardized JSON files

Creates:
1. mmlu_questions.json - 10 questions per MMLU subject (formatted)
2. truthfulqa_questions.json - TruthfulQA questions (formatted)
3. emobench_questions.json - 50 EmoBench questions (formatted)
4. eqbench_questions.json - 50 EQ-Bench questions (formatted)

All formatted to the same structure:
{
    "id": "unique_id",
    "question": "the question text",
    "source": "dataset_name",
    "subject": "subject_area",
    "correct_answer": "ground truth",
    "answer_choices": ["A", "B", "C", "D"] (if applicable)
}
"""

import json
from datasets import load_dataset
import os

os.environ["HF_DATASETS_OFFLINE"] = "1"

# ============================================================================
# MMLU - 10 questions per subject
# ============================================================================

def load_mmlu_questions(questions_per_subject=10):
    """Load MMLU questions with multiple choice answers"""
    
    # Selected subjects across different domains
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
    
    # Group by subject (only selected subjects)
    subjects = {}
    for example in mmlu:
        subject = example["subject"]
        if subject in SELECTED_SUBJECTS:
            if subject not in subjects:
                subjects[subject] = []
            subjects[subject].append(example)
    
    # Take first N from each subject
    formatted_questions = []
    for subject, examples in subjects.items():
        for i, example in enumerate(examples[:questions_per_subject]):
            # Format choices as A, B, C, D
            choices_dict = {
                "A": example["choices"][0],
                "B": example["choices"][1],
                "C": example["choices"][2],
                "D": example["choices"][3],
            }
            
            # Get correct answer letter
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

# ============================================================================
# TruthfulQA
# ============================================================================

def load_truthfulqa_questions(num_questions=50):
    """Load TruthfulQA questions"""
    
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

# ============================================================================
# EmoBench
# ============================================================================

def load_emobench_questions(num_questions=50):
    """Load EmoBench questions - format scenario as question"""
    
    print("Loading EmoBench...")
    emobench = load_dataset("SahandSab/EmoBench", "emotional_understanding", split="train", download_mode="reuse_cache_if_exists")
    
    formatted_questions = []
    for i, example in enumerate(emobench):
        if i >= num_questions:
            break
        
        # Format scenario as a question
        question_text = f"Read this scenario and identify the emotion {example['subject']} is feeling:\n\n{example['scenario']}\n\nWhat emotion is {example['subject']} experiencing?"
        
        # Format choices as dict
        choices_dict = {}
        for j, choice in enumerate(example["emotion_choices"]):
            letter = chr(65 + j)  # A, B, C, D, E, F...
            choices_dict[letter] = choice
        
        # Find correct answer letter
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

# ============================================================================
# EQ-Bench
# ============================================================================

def load_eqbench_questions(num_questions=50):
    """Load EQ-Bench questions - extract dialogue and emotions"""
    
    print("Loading EQ-Bench...")
    eqbench = load_dataset("pbevan11/EQ-Bench", split="validation", download_mode="reuse_cache_if_exists")
    
    formatted_questions = []
    for i, example in enumerate(eqbench):
        if i >= num_questions:
            break
        
        # The prompt contains the full dialogue and question
        # We'll use it as-is since it's already well-formatted
        
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

# ============================================================================
# Save to JSON
# ============================================================================

def save_questions():
    """Load all datasets and save to JSON files"""
    
    output_dir = "question_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load each dataset
    mmlu_questions = load_mmlu_questions(questions_per_subject=10)
    truthfulqa_questions = load_truthfulqa_questions(num_questions=50)
    emobench_questions = load_emobench_questions(num_questions=50)
    eqbench_questions = load_eqbench_questions(num_questions=50)
    
    # Save to JSON
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
    
    # Also create a combined dataset (excluding MMLU for separate run)
    combined_questions = truthfulqa_questions + emobench_questions + eqbench_questions
    combined_filepath = os.path.join(output_dir, "combined_150_questions.json")
    with open(combined_filepath, "w") as f:
        json.dump(combined_questions, f, indent=2)
    print(f"\n✓ Saved {len(combined_questions)} combined questions to {combined_filepath}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"MMLU: {len(mmlu_questions)} questions")
    print(f"TruthfulQA: {len(truthfulqa_questions)} questions")
    print(f"EmoBench: {len(emobench_questions)} questions")
    print(f"EQ-Bench: {len(eqbench_questions)} questions")
    print(f"Combined (no MMLU): {len(combined_questions)} questions")
    print(f"{'='*60}")

if __name__ == "__main__":
    save_questions()