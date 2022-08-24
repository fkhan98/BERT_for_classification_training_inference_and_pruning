# from transformers import XLMRobertaForSequenceClassification,XLMRobertaTokenizer
import torch
import pandas as pd

from textpruner import summary, TransformerPruner, TransformerPruningConfig, GeneralConfig
from transformers import BertForSequenceClassification, BertTokenizer
from model import BertClassifier
from dataset import Dataset
from torch.utils.data import DataLoader


model_path = "./saved_model"

model = BertForSequenceClassification.from_pretrained(model_path,num_labels=5)
tokenizer = BertTokenizer.from_pretrained(model_path)

print("Before pruning:")
print(summary(model))

general_config = GeneralConfig(use_device='auto',output_dir='./pruned_models')

transformer_pruning_config = TransformerPruningConfig(
    target_ffn_size=512, target_num_of_heads=6, 
    pruning_method='iterative',n_iters=10)

print('apatotoh okay after config defination')
df = pd.read_csv('./bbc-text.csv')
data = Dataset(df)
dataloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
print('dataloader:')
print(data)
pruner = TransformerPruner(model,transformer_pruning_config=transformer_pruning_config
                           ,general_config=general_config) 
print('lastly shob ok')
pruner.prune(dataloader=dataloader, save_model=True)


print("After pruning:")
print(summary(model))
