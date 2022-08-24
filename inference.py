import torch

from transformers import BertForSequenceClassification, BertTokenizer
from model import BertClassifier
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "./saved_model"

model = BertForSequenceClassification.from_pretrained(model_path,num_labels=5)
model.to(device)
print(model)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_text = 'tv future in the hands of viewers with home theatre systems  plasma high-definition tvs  and digital video recorders moving into the living room'

input = tokenizer(example_text,padding='max_length', max_length = 250, 
                       truncation=True, return_tensors="pt")

mask = input['attention_mask'].to(device)
input_id = input['input_ids'].squeeze(1).to(device)
with torch.no_grad():
    output = model(input_id, mask)

print(output)

