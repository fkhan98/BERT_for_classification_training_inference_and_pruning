# Preparing BERT for classification task
On top of the bert-base-cased model we add a dense layer with 768 input nodes and 5 output nodes. The details of the model are inside the model.py script.

# Trainig BERT for classification task
The model which was taken from hugging face was fine tuned on the bbc-text dataset available on Kaggle. The details of the training script are inside train.py.

# Pruning the trained model
The Textpruner library(https://github.com/airaria/TextPruner/) was used to prune the trained model to decrease it's size and to also increase infernece speed. The model pruning is implemented in prune.py script and after pruning comparisions with before and after pruned models are done in comparision_after_pruning.py script. Significant improvements in space and inference time is noticed.
# Resources 
-> https://github.com/airaria/TextPruner/
-> https://blog.51cto.com/u_14156307/5274012
-> https://zhuanlan.zhihu.com/p/469103382
