from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
import pandas as pd
import time
import re
import itertools
import os
import argparse
from tqdm import tqdm

# model
from model_arq import get_tokenizer, BETOReviewsClassifier
from data_manager import data_loader

# config
from useful import Config

TRAIN_DATA_PATH = "data/train.csv"

def get_training_data():
    df = pd.read_csv(TRAIN_DATA_PATH)
    df = df[["review","target"]]
    return df

# Training and validation iteration functions

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    print("### Training Model ###")
    model = model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, total=len(data_loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch['target'].to(device)

        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, target)
        correct_predictions += torch.sum(preds == target)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    print("### Evaluate Model ###")
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)

            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, target)
            correct_predictions += torch.sum(preds == target)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def get_args(config):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version-model",
        type=str,
        required=True,
        default='0',
        help="Version of model"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        default=config.epochs,
        help="number of epochs"
    )

    parser.add_argument(
        "--n-classes",
        type=int,
        required=False,
        default=config.n_classes,
        help="number of classes"
    )

    parser.add_argument(
        "--max-length-tokens",
        type=int,
        required=False,
        default=config.max_length_tokens,
        help="Max length of the tokens to be used"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=config.batch_size,
        help="BATCH_SIZE for training"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        required=False,
        default=config.random_seed,
        help="random seed for to start training"
    )

    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":

    config = Config()

    args = get_args(config)

    # Initialize
    RANDOM_SEED = args.random_seed
    MAX_LEN = args.max_length_tokens
    BATCH_SIZE = args.batch_size
    NCLASSES = args.n_classes
    EPOCHS = args.epochs
    VERSION_MODEL = args.version_model

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"### Device to use: {device} ###")

    tokenizer = get_tokenizer()

    model = BETOReviewsClassifier(NCLASSES)
    model = model.to(device)

    print("### Getting training data ###")
    df = get_training_data()

    print(f"### Amount of data to train: {df.shape[0]} ###")

    # split data and create data loaders
    x_train, x_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    train_data_loader = data_loader(x_train, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = data_loader(x_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # Training setting

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    # TRAIN!

    start_time = time.time()

    for epoch in range(EPOCHS):
        start_batch_time = time.time()

        print(f'Epoch {epoch+1} out of {EPOCHS}')
        print('---------------------------------')
        train_acc, train_loss = train_model(
            model, train_data_loader, loss_fn, optimizer, device, scheduler, len(x_train)
        )
        test_acc, test_loss = eval_model(
            model, test_data_loader, loss_fn, device, len(x_test)
        )
        end_batch_time = time.time()
        process_batch_time = round((end_batch_time - start_batch_time) / 60, 2)

        print(f'Training loss: {train_loss}, accuracy: {train_acc}')
        print(f'Validation loss: {test_loss}, accuracy: {test_acc}')
        print(f'Process batch time: {process_batch_time} minutes.')
        print('')

    end_time = time.time()
    process_time = round((end_time - start_time) / 60, 2)

    print(f'Fitting model with {EPOCHS} epochs took {process_time} minutes.')

    torch.save(model, f'/usr/src/models/review-clf-v{VERSION_MODEL}.pth')
    print(f"MODEL SAVED: {os.listdir('/usr/src/models/')}")

    torch.save(model.state_dict(), f'/usr/src/models/weights/review-weights-v{VERSION_MODEL}')
    print(f"WEIGHTS SAVED: {os.listdir('/usr/src/models/weights/')}")
