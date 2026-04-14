from datasets import load_dataset, get_dataset_config_names

# mmlu = load_dataset("cais/mmlu", "all", download_mode="force_redownload")
# truthfulqa = load_dataset("truthful_qa", "generation", download_mode="force_redownload")

# eq_bench = load_dataset("pbevan11/EQ-Bench", split="validation", trust_remote_code=True)
# print(eq_bench.column_names)
# print(eq_bench[0])

# print(f"EQ-Bench splits: {eq_bench}")

print("========= MMLU =========")
mmlu = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True, download_mode="reuse_cache_if_exists")
print(mmlu.column_names)
print(mmlu[0])
print(mmlu[1])
print(mmlu[2])
print(f"MMLU splits: {mmlu}")

subsets = get_dataset_config_names("cais/mmlu")
print(subsets)


print("\n========= TruthfulQA =========")
truthfulqa = load_dataset("truthful_qa", "generation", split="validation", trust_remote_code=True, download_mode="reuse_cache_if_exists")
print(truthfulqa.column_names)
print(truthfulqa[0])
print(f"TruthfulQA splits: {truthfulqa}")

print("\n========= EQ-Bench =========")
eq_bench = load_dataset("pbevan11/EQ-Bench", split="validation", trust_remote_code=True, download_mode="reuse_cache_if_exists")
print(eq_bench.column_names)
print(eq_bench[0])
print(f"EQ-Bench splits: {eq_bench}")

print("\n========= EmoBench =========")
emobench = load_dataset("SahandSab/EmoBench", "emotional_understanding", split="train", trust_remote_code=True)
print(emobench.column_names)
print(emobench[0])
print(f"EmoBench splits: {emobench}")

# print(f"EmoBench splits: {emobench}")

# from datasets import get_dataset_split_names

# print(get_dataset_split_names("truthful_qa", "generation"))