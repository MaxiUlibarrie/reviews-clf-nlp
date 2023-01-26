import torch
from torch.utils.data import Dataset, DataLoader

# DATASET

class ReviewsDataset(Dataset):

    def __init__(self, reviews, target, tokenizer, max_len):
        self.reviews = reviews
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.target[item]

        encoding = self.tokenizer.encode_plus(
            review,
            max_length = self.max_len,
            truncation = True,
            add_special_tokens = True,
            return_token_type_ids = False,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        return {
            'review': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.long)
        }

# DATA LOADER

def data_loader(df, tokenizer, max_len, batch_size):

    dataset = ReviewsDataset(
        reviews = df.review.to_numpy(),
        target = df.target.to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=4)
