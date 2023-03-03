from transformers import BertModel, BertTokenizer
from torch import nn

# https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased
# PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
PRE_TRAINED_MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'

def beto_tokenizer():
    return BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

class BETOReviewsClfModel(nn.Module):

    def __init__(self, n_classes):
        super(BETOReviewsClfModel, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        cls_output = bert_output[1] 
        t = self.drop(cls_output)
        t = self.relu(t)
        t = self.fc(t)
        output = self.softmax(t)

        return output
