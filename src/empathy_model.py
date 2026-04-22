from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers
import torch
import torch.nn.functional as F
import json
import pandas as pd


class EmpathyModel:
    def __init__(self):
        model_name = "bdotloh/roberta-base-empathy"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()


    def predict_empathy(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128) 
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        scores = logits.tolist()
        return scores   
