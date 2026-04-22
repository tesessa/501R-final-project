from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class EmotionModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("duelker/samo-goemotions-deberta-v3-large", use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained("duelker/samo-goemotions-deberta-v3-large") 
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ] 

    def predict_emotions(self, text, threshold=0.4):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = torch.sigmoid(logits).squeeze(0)
        predictions = {}
        for i, emotion in enumerate(self.emotion_labels):
            predictions[emotion] = {
                'probability': float(probabilities[i]),
                'predicted': probabilities[i] > threshold
            }
        
        return predictions