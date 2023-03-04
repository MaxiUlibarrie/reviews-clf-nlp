import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
import pandas as pd
import time
import os
from tqdm import tqdm

# model
from beto_reviews_clf import BETOReviewsClfModel, beto_tokenizer

class TrainModel():

    def __init__(self, random_seed, max_length_tokens, batch_size, n_classes, epochs, version_model):
        self.random_seed = random_seed
        self.max_length_tokens = max_length_tokens
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.epochs = epochs
        self.version_model = version_model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = beto_tokenizer()
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        print(f"### Device to use: {self.device} ###")

        self.model = BETOReviewsClfModel(self.n_classes)
        self.model = self.model.to(self.device)

    def train(self):

        train_df = self.get_training_data()

        # split data and create data loaders
        x_train, x_test = train_test_split(train_df, test_size=0.2, random_state=self.random_seed)

        train_data_loader = data_loader(x_train, self.tokenizer, self.max_length_tokens, self.batch_size)
        test_data_loader = data_loader(x_test, self.tokenizer, self.max_length_tokens, self.batch_size)

        # Training setting

        optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(train_data_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = 0,
            num_training_steps = total_steps
        )
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        # TRAIN!

        start_time = time.time()

        for epoch in range(self.epochs):
            start_batch_time = time.time()

            print(f'Epoch {epoch+1} out of {self.epochs}')
            print('---------------------------------')
            train_acc, train_loss = self.train_model(
                train_data_loader, loss_fn, optimizer, scheduler, len(x_train)
            )
            test_acc, test_loss = self.eval_model(
                test_data_loader, loss_fn, len(x_test)
            )
            end_batch_time = time.time()
            process_batch_time = round((end_batch_time - start_batch_time) / 60, 2)

            print(f'Training loss: {train_loss}, accuracy: {train_acc}')
            print(f'Validation loss: {test_loss}, accuracy: {test_acc}')
            print(f'Process batch time: {process_batch_time} minutes.')
            print('')

        end_time = time.time()
        process_time = round((end_time - start_time) / 60, 2)

        print(f'Fitting model with {self.epochs} epochs took {process_time} minutes.')


        path_new_model = os.environ.get("NEW_MODEL_PATH")
        # saving model
        torch.save(self.model, f'{path_new_model}/review-clf-v{self.version_model}.pth')

        # saving weigths
        torch.save(self.model.state_dict(), f'{path_new_model}/weights/review-weights-v{self.version_model}')

    def get_training_data(self):
        df = pd.read_csv(os.environ.get("TRAIN_DATA"))
        df = df[["review","target"]]
        return df

    # Training and validation iteration functions

    def train_model(self, data_loader, loss_fn, optimizer, scheduler, n_examples):
        print("### Training Model ###")
        self.model = self.model.train()
        losses = []
        correct_predictions = 0

        for batch in tqdm(data_loader, total=len(data_loader)):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target = batch['target'].to(self.device)

            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, target)
            correct_predictions += torch.sum(preds == target)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return correct_predictions.double() / n_examples, np.mean(losses)


    def eval_model(self, data_loader, loss_fn, n_examples):
        print("### Evaluate Model ###")
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, total=len(data_loader)):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target = batch['target'].to(self.device)

                outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, target)
                correct_predictions += torch.sum(preds == target)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)
    
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
