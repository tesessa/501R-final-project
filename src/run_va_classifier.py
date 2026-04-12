import json

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
import numpy as np
import prompts

high_ars_pos_val_prompts = {
    "excited": prompts.EXCITED_PROMPT,
    "joyful": prompts.JOYFUL_PROMPT,
    "amused": prompts.AMUSED_PROMPT,
    "enthusiastic": prompts.ENTHUSIASTIC_PROMPT,
    "excited_user": prompts.EXCITED_USER_PROMPT,
    "joyful_user": prompts.JOYFUL_USER_PROMPT,
    "amused_user": prompts.AMUSED_USER_PROMPT,
    "enthusiastic_user": prompts.ENTHUSIASTIC_USER_PROMPT
}

high_ars_neg_val_prompts = {
    "angry": prompts.ANGRY_PROMPT,
    "annoyed": prompts.ANNOYED_PROMPT,
    "afraid": prompts.AFRAID_PROMPT,
    "disgusted": prompts.DISGUSTED_PROMPT,
    "angry_user": prompts.ANGRY_USER_PROMPT,
    "annoyed_user": prompts.ANNOYED_USER_PROMPT,
    "afraid_user": prompts.AFRAID_USER_PROMPT,
    "disgusted_user": prompts.DISGUSTED_USER_PROMPT
}

low_ars_pos_val_prompts = {
    "content": prompts.CONTENT_PROMPT,
    "relief": prompts.RELIEF_PROMPT,
    "satisfied": prompts.SATISFIED_PROMPT,
    "grateful": prompts.GRATEFUL_PROMPT,
    "content_user": prompts.CONTENT_USER_PROMPT,
    "relief_user": prompts.RELIEF_USER_PROMPT,
    "satisfied_user": prompts.SATISFIED_USER_PROMPT,
    "grateful_user": prompts.GRATEFUL_USER_PROMPT
}

low_ars_neg_val_prompts = {
    "sad": prompts.SAD_PROMPT,
    "lonely": prompts.LONELY_PROMPT,
    "bored": prompts.BORED_PROMPT,
    "fatigued": prompts.FATIGUED_PROMPT,
    "sad_user": prompts.SAD_USER_PROMPT,
    "lonely_user": prompts.LONELY_USER_PROMPT,
    "bored_user": prompts.BORED_USER_PROMPT,
    "fatigued_user": prompts.FATIGUED_USER_PROMPT
}

neutral_prompts = {
    "neutral": prompts.NEUTRAL_PROMPT,
    "neutral2": prompts.NEUTRAL_PROMPT2,
    "focused": prompts.FOCUSED_PROMPT,
    "unaffected": prompts.UNAFFECTED_PROMPT,
    "indifferent": prompts.INDIFFERENT_PROMPT,
    "neutral_user": prompts.NEUTRAL_USER_PROMPT,
    "focused_user": prompts.FOCUSED_USER_PROMPT,
    "focused_user2": prompts.FOCUSED_USER_PROMPT2,
    "unaffected_user": prompts.UNAFFECTED_USER_PROMPT,
    "indifferent_user": prompts.INDIFFERENT_USER_PROMPT
}

class VAPredictor:
    def __init__(self, model_dir, use_cuda=True):
        # Handle CUDA error - use CPU if CUDA fails
        if use_cuda and torch.cuda.is_available():
            try:
                self.device = torch.device('cuda')
                # Test CUDA
                torch.zeros(1).to(self.device)
            except RuntimeError:
                print("⚠ CUDA unavailable, using CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        
        # Load checkpoint
        checkpoint = torch.load(
            f"{model_dir}/pytorch_model.bin",
            map_location='cpu',
            weights_only=False
        )
        
        # Load config
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        
        # Build model with correct architecture
        self.model = self._build_model(config)
        
        # Load weights
        missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
        
        # Filter out unimportant missing/unexpected keys
        missing = [k for k in missing if 'position_ids' not in k]
        unexpected = [k for k in unexpected if 'position_ids' not in k]
        
        if missing:
            print(f"⚠ Missing keys: {missing}")
        if unexpected:
            print(f"⚠ Unexpected keys: {unexpected}")
        
        if not missing and not unexpected:
            print(f"✓ Model loaded successfully - all weights matched!")
        else:
            print(f"✓ Model loaded (some keys didn't match, but this may be okay)")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _build_model(self, config):
        """Build model matching checkpoint structure EXACTLY"""
        class VAModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                # RoBERTa encoder (no pooling layer)
                self.roberta = AutoModel.from_config(config, add_pooling_layer=False)
                
                # Classifier - must match checkpoint keys exactly:
                # checkpoint has: classifier.dense.weight/bias and classifier.out_proj.weight/bias
                class ClassifierHead(nn.Module):
                    def __init__(self, hidden_size):
                        super().__init__()
                        self.dense = nn.Linear(hidden_size, hidden_size)
                        self.out_proj = nn.Linear(hidden_size, 2)
                    
                    def forward(self, features):
                        x = self.dense(features)
                        x = torch.tanh(x)
                        x = self.out_proj(x)
                        return x
                
                self.classifier = ClassifierHead(config.hidden_size)
            
            def forward(self, input_ids, attention_mask):
                outputs = self.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # Get [CLS] token
                sequence_output = outputs.last_hidden_state
                pooled = sequence_output[:, 0, :]  # [CLS] token
                
                logits = self.classifier(pooled)
                return logits
        
        return VAModel(config)
    
    def predict(self, texts):
        """Predict valence and arousal on 0-1 scale"""
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply sigmoid to convert logits to 0-1 range
            predictions = torch.sigmoid(outputs).cpu().numpy()
        
        results = [(float(v), float(a)) for v, a in predictions]
        
        return results[0] if single else results
    
    def predict_with_scales(self, texts):
        """Predict and return in multiple scales"""
        if isinstance(texts, str):
            single = True
            texts = [texts]
        else:
            single = False

        predictions = self.predict(texts)
        if single:
            # predictions is a single tuple (v, a)
            v, a = predictions[0]
            result = {
                'valence_0_1': v,
                'arousal_0_1': a,
                'valence_1_9': 1 + (v * 8),
                'arousal_1_9': 1 + (a * 8),
                'valence_neg1_pos1': (v * 2) - 1,
                'arousal_neg1_pos1': (a * 2) - 1,
            }
            return result
        else:
            # predictions is a list of tuples [(v1, a1), (v2, a2), ...]
            results = []
            for v, a in predictions:
                result = {
                    'valence_0_1': v,
                    'arousal_0_1': a,
                    'valence_1_9': 1 + (v * 8),
                    'arousal_1_9': 1 + (a * 8),
                    'valence_neg1_pos1': (v * 2) - 1,
                    'arousal_neg1_pos1': (a * 2) - 1,
                }
                results.append(result)
            return results

    def predict_all_scales(self, text):
        """Test different normalization strategies"""
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs).cpu().numpy()[0]
            raw_v, raw_a = outputs[0], outputs[1]
        
        print(f"\nText: {text}")
        print(f"Raw outputs: V={raw_v:.3f}, A={raw_a:.3f}")
        print("\nNormalization strategies:")
        
        # Strategy 1: 1-5 scale
        v_1_5 = np.clip((raw_v - 1) / 4, 0, 1)
        a_1_5 = np.clip((raw_a - 1) / 4, 0, 1)
        print(f"  1-5 scale: V={v_1_5:.3f}, A={a_1_5:.3f}")
        
        # Strategy 2: 1-9 scale
        v_1_9 = np.clip((raw_v - 1) / 8, 0, 1)
        a_1_9 = np.clip((raw_a - 1) / 8, 0, 1)
        print(f"  1-9 scale: V={v_1_9:.3f}, A={a_1_9:.3f}")
        
        # Strategy 3: tanh
        v_tanh = (np.tanh(raw_v) + 1) / 2
        a_tanh = (np.tanh(raw_a) + 1) / 2
        print(f"  tanh:      V={v_tanh:.3f}, A={a_tanh:.3f}")
        
        # Strategy 4: sigmoid (what we tried before)
        v_sig = 1 / (1 + np.exp(-raw_v))
        a_sig = 1 / (1 + np.exp(-raw_a))
        print(f"  sigmoid:   V={v_sig:.3f}, A={a_sig:.3f}")


# Usage
model_dir = "/home/tessa343/classes/501R-final-project/src/va_model"

predictor = VAPredictor(model_dir, use_cuda=False)


# Test on clear examples
# test_texts = [
#     "I am so depressed, my grandma died and I tried to kill myself yesterday",
#     "I just got a promotion at work and I'm feeling fantastic!",
#     "I am extremely anxious about my upcoming exams.",
#     "I feel so relaxed and content right now.",
#     "I am heartbroken and devastated after the breakup.",
# ]

# print("\n" + "="*80)
# print("Testing Valence-Arousal Predictions")
# print("="*80)

# for text in test_texts:
#     result = predictor.predict_with_scales(text)
    
#     print(f"\nText: {text}")
#     print(f"  Valence: {result['valence_0_1']:.3f} (0-1) = {result['valence_1_9']:.2f} (1-9)")
#     print(f"  Arousal: {result['arousal_0_1']:.3f} (0-1) = {result['arousal_1_9']:.2f} (1-9)")





print("\n" + "="*80)
print("Testing Positive Valence High Arousal Predictions")
print("="*80)

ratings = {}

for emotion, text in high_ars_pos_val_prompts.items():
    result = predictor.predict_with_scales(text)
    
    print(f"\nEmotion: {emotion} \nText: {text}")
    print(f"  Valence: {result['valence_0_1']:.3f} (0-1) = {result['valence_1_9']:.2f} (1-9)")
    print(f"  Arousal: {result['arousal_0_1']:.3f} (0-1) = {result['arousal_1_9']:.2f} (1-9)")
    ratings[emotion] = {
        'text': text,
        'valence_0_1': result['valence_0_1'],
        'arousal_0_1': result['arousal_0_1'],
        'valence_1_9': result['valence_1_9'],
        'arousal_1_9': result['arousal_1_9']
    }
    # with open("ratings.txt", "a") as f:




print("\n" + "="*80)
print("Testing Negative Valence High Arousal Predictions")
print("="*80)

for emotion, text in high_ars_neg_val_prompts.items():
    result = predictor.predict_with_scales(text)
    
    print(f"\nEmotion: {emotion} \nText: {text}")
    print(f"  Valence: {result['valence_0_1']:.3f} (0-1) = {result['valence_1_9']:.2f} (1-9)")
    print(f"  Arousal: {result['arousal_0_1']:.3f} (0-1) = {result['arousal_1_9']:.2f} (1-9)")
    ratings[emotion] = {
        'text': text,
        'valence_0_1': result['valence_0_1'],
        'arousal_0_1': result['arousal_0_1'],
        'valence_1_9': result['valence_1_9'],
        'arousal_1_9': result['arousal_1_9']
    }


print("\n" + "="*80)
print("Testing Positive Valence Low Arousal Predictions")
print("="*80)

for emotion, text in low_ars_pos_val_prompts.items():
    result = predictor.predict_with_scales(text)
    
    print(f"\nEmotion: {emotion} \nText: {text}")
    print(f"  Valence: {result['valence_0_1']:.3f} (0-1) = {result['valence_1_9']:.2f} (1-9)")
    print(f"  Arousal: {result['arousal_0_1']:.3f} (0-1) = {result['arousal_1_9']:.2f} (1-9)")
    ratings[emotion] = {
        'text': text,
        'valence_0_1': result['valence_0_1'],
        'arousal_0_1': result['arousal_0_1'],
        'valence_1_9': result['valence_1_9'],
        'arousal_1_9': result['arousal_1_9']
    }


print("\n" + "="*80)
print("Testing Negative Valence Low Arousal Predictions")
print("="*80)

for emotion, text in low_ars_neg_val_prompts.items():
    result = predictor.predict_with_scales(text)
    
    print(f"\nEmotion: {emotion} \nText: {text}")
    print(f"  Valence: {result['valence_0_1']:.3f} (0-1) = {result['valence_1_9']:.2f} (1-9)")
    print(f"  Arousal: {result['arousal_0_1']:.3f} (0-1) = {result['arousal_1_9']:.2f} (1-9)")
    ratings[emotion] = {
        'text': text,
        'valence_0_1': result['valence_0_1'],
        'arousal_0_1': result['arousal_0_1'],
        'valence_1_9': result['valence_1_9'],
        'arousal_1_9': result['arousal_1_9']
    }

print("\n" + "="*80)
print("Testing Neutral Prompts Predictions")
print("="*80)

for emotion, text in neutral_prompts.items():
    result = predictor.predict_with_scales(text)
    
    print(f"\nEmotion: {emotion} \nText: {text}")
    print(f"  Valence: {result['valence_0_1']:.3f} (0-1) = {result['valence_1_9']:.2f} (1-9)")
    print(f"  Arousal: {result['arousal_0_1']:.3f} (0-1) = {result['arousal_1_9']:.2f} (1-9)")
    ratings[emotion] = {
        'text': text,
        'valence_0_1': result['valence_0_1'],
        'arousal_0_1': result['arousal_0_1'],
        'valence_1_9': result['valence_1_9'],
        'arousal_1_9': result['arousal_1_9']
    }


with open("ratings.json", "a") as f:
    json.dump(ratings, f, indent=4)
