import torch

from torch import nn
from transformers import BertModel
# from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertPreTrainedModel

    
class BertClassifier(BertPreTrainedModel):

    def __init__(self, config, dropout=0.5):

        super(BertClassifier, self).__init__(config)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        # self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1) 
        # self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(device)
    def forward(self, input_id, mask, label=None):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        # final_layer = self.relu(linear_output)
        final_layer = self.softmax(linear_output)
        
        if label == None:
            return final_layer
        else:
            loss = self.criterion(final_layer, label.long())
            return SequenceClassifierOutput(
                loss=loss,
                logits = final_layer,
            )