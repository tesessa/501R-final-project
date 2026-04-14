from datasets import load_dataset, get_dataset_config_names
import os
import json

# subjects = ['abstract_algebra', 'all', 'anatomy', 'astronomy', 'auxiliary_train', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
# subjects = ["abstract_algebra", "machine_learning", "formal_logic", "college_physics", "anatomy", "high_school_biology", "philosophy", "world_religions", "high_school_european_history", "econometrics", "business_ethics", "international_law", "professional_psychology"]
# # 130 mmlu questions
# mmlu_samples = {}

# for subject in subjects:
#     print(subject)
#     # skip "all" config if present
#     if subject == "all" or subject == "auxiliary_train":
#         continue

#     os.environ["HF_DATASETS_OFFLINE"] = "1"
#     os.environ["TRANSFORMERS_OFFLINE"] = "1"

#     ds = load_dataset(
#         "cais/mmlu",
#         subject,
#         split="test",
#         trust_remote_code=True,
#         download_mode="reuse_cache_if_exists"
#     )

#     # Take first 10 questions (or fewer if dataset is small)
#     mmlu_samples[subject] = ds.select(range(min(10, len(ds))))
    
# serializable_samples = {}

# for subject, ds in mmlu_samples.items():
#     cleaned = []
    
#     for ex in ds:
#         cleaned.append({
#             "question": ex["question"],
#             "choices": ex["choices"],
#             "answer": ex["answer"]
#         })
    
#     serializable_samples[subject] = cleaned

# with open("mmlu_samples.json", "w") as f:
#     json.dump(serializable_samples, f, indent=2)



# truthfulqa_samples = {}
# truthfulqa = load_dataset("truthful_qa", "generation", split="validation", trust_remote_code=True, download_mode="reuse_cache_if_exists")
# for i in range(50):
#     truthfulqa_samples[str(i+1)] = truthfulqa[i]

truthfulqa = load_dataset(
    "truthful_qa",
    "generation",   # or "generation"
    split="validation"   # truthfulQA uses validation, not test
)

samples = []

for ex in truthfulqa.select(range(50)):
    samples.append({
        "question": ex["question"],
        "answer": ex["correct_answers"],
        "inccorect_answers": ex["incorrect_answers"],
        "type": ex["type"],
        "category": ex["category"],
    })

with open("truthfulqa_50.json", "w") as f:
    json.dump(samples, f, indent=2)



        # for i, example in enumerate(truthfulqa):
        #     if len(questions) >= num_questions // 4:
        #         break
        #     questions.append({
        #         "id": f"truthfulqa_{i}",
        #         "question": example["question"],
        #         "source": "truthfulqa",
        #         "subject": "truthfulness",
        #     })
