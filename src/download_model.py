# # download_model.py
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "meta-llama/Llama-3.1-8B-Instruct"

# print("Downloading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# print("Downloading model...")
# model = AutoModelForCausalLM.from_pretrained(model_name)

# print("Done! Model cached.")

# download_datasets.py
# download_all_data.py
import os
os.environ["HF_DATASETS_OFFLINE"] = "0"  # Allow downloads

from datasets import load_dataset

datasets_info = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy"),
    "hellaswag": ("Rowan/hellaswag", None),
    "winogrande": ("allenai/winogrande", "winogrande_xl"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None),
}

for name, (dataset_path, config) in datasets_info.items():
    print(f"\n{'='*60}")
    print(f"Downloading {name}: {dataset_path}")
    print('='*60)
    
    try:
        if config:
            dataset = load_dataset(dataset_path, config, download_mode="force_redownload")
        else:
            dataset = load_dataset(dataset_path, download_mode="force_redownload")
        
        print(f"✓ Successfully downloaded {name}")
        print(f"  Splits: {list(dataset.keys())}")
        if 'test' in dataset:
            print(f"  Test samples: {len(dataset['test'])}")
    except Exception as e:
        print(f"✗ Failed to download {name}: {e}")

print("\n" + "="*60)
print("Download complete! Datasets cached in:")
print("~/.cache/huggingface/datasets/")
print("="*60)