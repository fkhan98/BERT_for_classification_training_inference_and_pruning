# from torch.optim import Adam
import torch
import numpy as np
import pandas as pd
import sys
import os

from tqdm import tqdm
from dataset import Dataset
from bert_classifier import BertClassifier
from transformers import BertConfig
from transformers import BertTokenizer

def train(model, train_data, val_data, learning_rate, epochs, batch_size):

    train, val = Dataset(train_data,'not_prune'), Dataset(val_data,'not_prune')

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    best_loss = sys.maxsize
    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            model.train()
            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask, train_label)
                
                batch_loss = output.loss
                total_loss_train += batch_loss.item()
                
                acc = (output.logits.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            model.eval() 
            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask, val_label)

                    batch_loss = output.loss
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.logits.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
               
                path = './saved_model'
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                
                # save_path = os.path.join(path,'best_model.pt')
                if (total_loss_val / len(val_data)) < best_loss:
                    print(f'validation loss decreased from {best_loss} to {total_loss_val / len(val_data)}, model being saved')
                    best_loss = total_loss_val / len(val_data)
                    model.save_pretrained(path)
                    # torch.save(model.state_dict(), save_path)
                    

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
            
EPOCHS = 11
config = BertConfig.from_json_file('./config.json')
model = BertClassifier(config = config)
LR = 1e-6
batch_size = 32

df_train = pd.read_csv('./train.csv') 
df_val = pd.read_csv('./validation.csv') 
    
train(model, df_train, df_val, LR, EPOCHS, batch_size)
##saving tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.save_pretrained("./saved_model")