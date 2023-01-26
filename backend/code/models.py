from pydantic import BaseModel, constr
from typing import Optional
import os
import re
import torch
from transformers import BertTokenizer
from model_arq import get_tokenizer, BETOReviewsClassifier
from glob import glob

# config
from useful import Config

PATH_MODELS = "/usr/src/models"

class ReviewPL(BaseModel):
    review: Optional[constr(max_length=200)]

class Reviews_clf():

    def __init__(self):
        print("### LOADING MODEL ###")

        all_models = glob(PATH_MODELS + "/review-clf-v*.pth")
        last_model = sorted(all_models)[-1]
        self.clf = torch.load(last_model)

        print(f"Loaded Model: {last_model}")

        self.tokenizer = get_tokenizer()

        config = Config()
        self.MAX_LEN = config.max_length_tokens

        self.target_map = {
            0: 'NEGATIVO', 
            1: 'POSITIVO'
        }
        
        print("### MODEL LOADED SUCCESSFULLY ###")

    def __transform_input(self, review):
        text = str(review)
        text = text.upper()
        return text

    def classify_review(self, review):
        encoding_review = self.tokenizer.encode_plus(
            self.__transform_input(review),
            max_length = self.MAX_LEN,
            truncation = True,
            add_special_tokens = True,
            return_token_type_ids = False,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        
        input_ids = encoding_review['input_ids']
        attention_mask = encoding_review['attention_mask']
        output = self.clf(input_ids, attention_mask) 
        prediction = torch.argmax(output.flatten()).item()

        return self.target_map[prediction]
