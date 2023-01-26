from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased
# PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
PRE_TRAINED_MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'

# tokenizer

def get_tokenizer():
    print("### Getting Tokenizer ###")
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    return tokenizer

# MODEL!

class BETOReviewsClassifier(nn.Module):

    def __init__(self, n_classes):
        super(BETOReviewsClassifier, self).__init__()
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
        cls_output = bert_output[1] # passing the pooler_output/token classification
        t = self.drop(cls_output)
        t = self.relu(t)
        t = self.fc(t)
        output = self.softmax(t)

        return output
