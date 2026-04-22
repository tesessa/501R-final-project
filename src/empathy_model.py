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



# runs empathy model on input_file, need to make this more dynamic I guess?
# model_name = "bdotloh/roberta-base-empathy"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# model.eval()

# def predict_empathy(texts):
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits

#     #probabilities = torch.sigmoid(logits).squeeze(0)
#     scores = logits.tolist()
#     return scores    

# input_file = '/home/tessa343/classes/501R-final-project/src/results/final_run/judged_responses_new.json'

# with open(input_file, 'r') as f:
#     data = json.load(f)


# batch_size = 50

# for start in range(0, len(data), batch_size):
#     batch = data[start:start + batch_size]

#     input_texts = []
#     output_texts = []
#     valid_indices = []

#     # Build batch
#     for i, obj in enumerate(batch):
#         # if obj['emotion'] == 'control':
#         #     continue

#         input_texts.append(obj['emotional_prefix'])
#         output_texts.append(obj['response'])
#         valid_indices.append(i)

#     # Skip empty batch
#     if not input_texts:
#         continue

#     # Run model
#     input_scores = predict_empathy(input_texts)
#     output_scores = predict_empathy(output_texts)

#     # Write back results
#     for j, idx in enumerate(valid_indices):
#         data[start + idx]['input_empathy_score'] = input_scores[j][0]
#         data[start + idx]['input_distress_score'] = input_scores[j][1]
#         data[start + idx]['output_empathy_score'] = output_scores[j][0]
#         data[start + idx]['output_distress_score'] = output_scores[j][1]

#     print(f"Processed batch {start} to {start + batch_size}")
#     with open('/home/tessa343/classes/501R-final-project/src/results/final_run/judged_responses2.json', 'w') as f:
#         json.dump(data, f)

# input_texts = []
# output_texts = []
# batch_size = 50


# for index, obj in enumerate(data):
#     if obj['emotion'] == 'control':
#         pass
#     for i in range(batch_size):
#         input_texts.append(obj['emotion_prefix'])
#         output_texts.append(obj['response'])

#     # input_text = obj['emotion_prefix']
#     # output_text = obj['response']
#     input_empathy_scores = predict_empathy(input_texts)
#     output_empathy_scores = predict_empathy(output_texts)
#     print("Input Empathy Scores:", input_empathy_scores)
#     print("Output Empathy Scores:", output_empathy_scores)

#     for i in range(batch_size):
#         data[index]['input_empathy_score'] = input_empathy_scores[i][0]
#         data[index]['input_distress_scores'] = input_empathy_scores[i][1]
#         data[index]['output_empathy_score'] = output_empathy_scores[i][0]
#         data[index]['output_distress_score'] = output_empathy_scores[i][1]
#     with open('judged_responses2.json', 'w') as f:
#         json.dump(data, f, indent=4)


# input_texts = []
# output_texts = []
# for i in range(4):
#     input_file = f'results_{i}.json'
#     with open(input_file, 'r') as f:
#         data = json.load(f)
    
#     input_texts.append(data['input_text'])
#     output_texts.append(data['model_response'])

# input_empathy_scores = predict_empathy(input_texts)
# output_empathy_scores = predict_empathy(output_texts)
# print("Input Empathy Scores:", input_empathy_scores)
# print("Output Empathy Scores:", output_empathy_scores)
# for i in range(4):
#     input_file = f'results_{i}.json'
#     with open(input_file, 'r') as f:
#         data = json.load(f)
#     data['input_empathy_scores'] = input_empathy_scores[i][0]
#     data['input_distress_scores'] = input_empathy_scores[i][1]
#     data['output_empathy_scores'] = output_empathy_scores[i][0]
#     data['output_distress_scores'] = output_empathy_scores[i][1]
#     with open(input_file, 'w') as f:
#         json.dump(data, f, indent=4)