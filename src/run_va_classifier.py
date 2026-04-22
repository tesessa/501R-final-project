# import json

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
import numpy as np


class VAPredictor:
    def __init__(self, model_dir="/home/tessa343/classes/501R-final-project/src/va_model", use_cuda=True):
        if use_cuda and torch.cuda.is_available():
            try:
                self.device = torch.device('cuda')
                torch.zeros(1).to(self.device)
            except RuntimeError:
                print("⚠ CUDA unavailable, using CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        
        checkpoint = torch.load(
            f"{model_dir}/pytorch_model.bin",
            map_location='cpu',
            weights_only=False
        )
        
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        
        self.model = self._build_model(config)
        
        missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)

        missing = [k for k in missing if 'position_ids' not in k]
        unexpected = [k for k in unexpected if 'position_ids' not in k]
        
        if missing:
            print(f"⚠ Missing keys: {missing}")
        if unexpected:
            print(f"⚠ Unexpected keys: {unexpected}")
        
        if not missing and not unexpected:
            print(f"✓ Valence Arousal Model loaded successfully")
        else:
            print(f"✓ Model loaded (some keys didn't match, but this may be okay)")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _build_model(self, config):
        class VAModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.roberta = AutoModel.from_config(config, add_pooling_layer=False)

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
                sequence_output = outputs.last_hidden_state
                pooled = sequence_output[:, 0, :]  
                
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
            predictions = torch.sigmoid(outputs).cpu().numpy() # apply sigmiod like in paper
        
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