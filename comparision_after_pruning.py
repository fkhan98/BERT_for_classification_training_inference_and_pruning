import torch

from transformers import BertForSequenceClassification, BertTokenizer
from textpruner import summary
from textpruner import inference_time

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

##comparing before and after pruning model size
unpruned_model_path = "./saved_model"
unpruned_model = BertForSequenceClassification.from_pretrained(unpruned_model_path,num_labels=5)
unpruned_model.to(device)
unpruned_model.eval()
print("Before pruning:")
print(summary(unpruned_model))

pruned_model_path = "./pruned_models/pruned_H6.0F512"
pruned_model = BertForSequenceClassification.from_pretrained(pruned_model_path,num_labels=5)
pruned_model.to(device)
pruned_model.eval()
print("After pruning:")
print(summary(pruned_model))

##comparing before and after pruning inference time
example_text = 'tv future in the hands of viewers with home theatre systems  plasma high-definition tvs  and digital video recorders moving into the living room'
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
input = tokenizer(example_text,padding='max_length', max_length = 250, 
                       truncation=True, return_tensors="pt")
dummmy_input = (input['input_ids'].squeeze(1).to(device), input['attention_mask'].to(device))

print("Inference time before pruning:")
inference_time(unpruned_model, dummmy_input)

print("Inference time after pruning:")
inference_time(pruned_model, dummmy_input)




