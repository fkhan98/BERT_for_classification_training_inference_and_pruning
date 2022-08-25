# Preparing BERT for classification task
On top of the bert-base-cased model we add a dense layer with 768 input nodes and 5 output nodes. The BertClassifier class was written to replicate the forward function of the transformer based classifer models on huggingface. The details of the model are inside the bert_classifier.py script.

# Trainig BERT for classification task
The model which was taken from hugging face was fine tuned on the bbc-text dataset available on Kaggle. The dataset is also available in this repo. To make the train/valid/test split call the make_dataset_split.py. The details of the training script are inside train.py. The details of the dataloader used to train the classifier model is inside the dataset.py script.

# Pruning the trained model
The Textpruner library(https://github.com/airaria/TextPruner/) was used to prune the trained model to decrease it's size and to also increase infernece speed, keeping the accuracy as close as possible to the unpruned model. The model pruning is implemented in prune.py script and this script also displays the results(model size, inference time, accuracy) before and after pruning. 

# Steps to recreate the results
1. Run the make_dataset_split.py to create the train/valid/test split.
2. Run the train.py script to train the model. The model and tokenizer will be saved inside ./saved_model
3. Run the prune.py script to prune the trained model inside ./saved_model and the pruned models will be saved inside ./pruned_models. This script will also visualize the results of before and after pruning.

# Detailed explanation of how pruning and textpruner works
* https://docs.google.com/document/d/1WI9r3kFv_USbSU2XrAj426wG78OjKS1yjbLr2TXa-rE/edit?usp=sharing

# Resources 
* https://aclanthology.org/2022.acl-demo.4/ (link of the paper of TextPruner)
* https://github.com/airaria/TextPruner/
* https://blog.51cto.com/u_14156307/5274012
* https://zhuanlan.zhihu.com/p/469103382
