from torch import nn
from transformers import BertModel
from transformers.modeling_utils import PreTrainedModel

# class BertClassifier(nn.Module):

#     def __init__(self, dropout=0.5):

#         super(BertClassifier, self).__init__()

#         self.bert = BertModel.from_pretrained('bert-base-cased')
#         self.dropout = nn.Dropout(dropout)
#         self.linear = nn.Linear(768, 5)
#         self.relu = nn.ReLU()

#     def forward(self, input_id, mask):

#         _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
#         dropout_output = self.dropout(pooled_output)
#         linear_output = self.linear(dropout_output)
#         final_layer = self.relu(linear_output)

#         return final_layer
class BertClassifier(PreTrainedModel):

    def __init__(self, config, dropout=0.5):

        super(BertClassifier, self).__init__(config)

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1) 
        # self.config = config

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        # final_layer = self.relu(linear_output)
        final_layer = self.softmax(linear_output)

        return final_layer