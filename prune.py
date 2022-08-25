import torch
import pandas as pd

from transformers import BertConfig
from bert_classifier import BertClassifier
from textpruner import summary, TransformerPruner, TransformerPruningConfig, GeneralConfig, inference_time
from transformers import BertTokenizer
from bert_classifier import BertClassifier
from dataset import Dataset
from tqdm import tqdm

def measure_accuracy(model, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test = Dataset(test_dataset,flag = 'not_prune')
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
    
    total_acc = 0
    count = 0
    for test_input, test_label in tqdm(test_dataloader):
        count += 1
        test_label = test_label.to(device)
        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)
        with torch.no_grad():
            output = model(input_id, mask)
        
        acc = (output.argmax(dim=1) == test_label).sum().item()
        total_acc += acc
    return (total_acc/count)*100



df_test = pd.read_csv('./test.csv') 
model_path = "./saved_model"
model = BertClassifier.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

example_text = 'tv future in the hands of viewers with home theatre systems  plasma high-definition tvs  and digital video recorders moving into the living room'
input = tokenizer(example_text,padding='max_length', max_length = 512, 
                       truncation=True, return_tensors="pt")
device = 'cpu'
dummmy_input = (input['input_ids'].squeeze(1).to(device), input['attention_mask'].to(device))
print("Before pruning:")
print(summary(model))
print("Inference time before pruning:")
inference_time(model, dummmy_input)
acc_before_pruning = measure_accuracy(model, df_test)
print("Accuracy before pruning: ", acc_before_pruning)

general_config = GeneralConfig(use_device='auto',output_dir='./pruned_models')

transformer_pruning_config = TransformerPruningConfig(
    target_ffn_size=1536, target_num_of_heads=10, 
    pruning_method='iterative',n_iters=32)

df = pd.read_csv('./train.csv')
data = Dataset(df)
dataloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
print('dataloader:')
print(data)
pruner = TransformerPruner(model,transformer_pruning_config=transformer_pruning_config
                           ,general_config=general_config) 
pruner.prune(dataloader=dataloader, save_model=True)


print("After pruning:")
print(summary(model))
print("Inference time after pruning:")
inference_time(model, dummmy_input)
acc_after_pruning = measure_accuracy(model, df_test)
print("Accuracy after pruning: ", acc_after_pruning)
