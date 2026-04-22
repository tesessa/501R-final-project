from datasets import load_dataset, get_dataset_config_names
# run this on login node before running experiments to make sure datasets/models are downloaded


# used mmlu, truthful_qa, eq_bench, emo_bench, llama_3_8B
# mmlu = load_dataset("cais/mmlu", "all", download_mode="force_redownload")
# truthfulqa = load_dataset("truthful_qa", "generation", download_mode="force_redownload")
# eq_bench = load_dataset("pbevan11/EQ-Bench", split="validation", trust_remote_code=True, download_mode="force_redownload")
# emobench = load_dataset("SahandSab/EmoBench", "emotional_understanding", split="train", trust_remote_code=True, download_mode="force_redownload")


mmlu = load_dataset("cais/mmlu", "all")
truthfulqa = load_dataset("truthful_qa", "generation")
eq_bench = load_dataset("pbevan11/EQ-Bench")
emobench = load_dataset("SahandSab/EmoBench", "emotional_understanding")
arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")
hellaswag = load_dataset("Rowan/hellaswag")
winogrande = load_dataset("allenai/winogrande", "winogrande_xl")
mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro")
truthfulqa_mc = load_dataset("truthful_qa", "multiple_choice")

# print(mmlu['test'][0])
# print(truthfulqa['validation'][0])
# print(eq_bench['validation'][0])
# print(emobench['train'][0])
# eq_bench = load_dataset("pbevan11/EQ-Bench", split="validation", trust_remote_code=True)
# print(eq_bench.column_names)
# print(eq_bench[0])

print(f"MMLU splits: {mmlu}")
print(f"\nTruthfulQA splits: {truthfulqa}")
print(f"\nEQ-Bench splits: {eq_bench}")
print(f"\nEmoBench splits: {emobench}")
print(f"\nArc-easy split: {arc_easy}")
print(f"\nArc challenge split: {arc_challenge}")
print(f"\nHellaswag split: {hellaswag}")
print(f"\nWino grande split: {winogrande}")
print(f"\nMMLU Pro split: {mmlu_pro}")
print(f"\ntruthfulqa multiple choice split: {truthfulqa_mc}")



# download model
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.1-8B-Instruct"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Done! Model cached.")