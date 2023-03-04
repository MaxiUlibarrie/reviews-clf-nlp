from pydantic import BaseModel, constr
from typing import Optional
import os
import torch
from glob import glob

from beto_reviews_clf import BETOReviewsClfModel, beto_tokenizer

# config
from config import Config

config = Config()

class ReviewPL(BaseModel):
    review: Optional[constr(max_length=200)]

class Reviews_clf():

    target_map = {
        0: 'NEGATIVO', 
        1: 'POSITIVO'
    }

    def __init__(self):
        print("### LOADING MODEL ###")

        models_path = os.environ.get("MODELS_PATH")
        model = glob(f"{models_path}/review-clf-v*.pth")[0]
        self.clf = torch.load(model)

        print(f"Loaded Model: {model}")

        self.tokenizer = beto_tokenizer()

        self.MAX_LEN = config.get.model.max_length_tokens
        
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

        return Reviews_clf.target_map[prediction]
