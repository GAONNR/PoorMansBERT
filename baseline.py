# The major part of the code is adopted from https://github.com/deepampatel/TwinBert

import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn import metrics
from torch import cuda, optim
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

df = pd.read_csv("data/train.csv")
df.head()


parser = argparse.ArgumentParser()
parser.add_argument('--drop', action='store_true')
parser.add_argument('--remove_layers', type=str, default='',
                    help="specify layer numbers to remove during finetuning e.g. 0,1,2 to remove first three layers")
args = parser.parse_args()


class SiameseNetworkDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.question1 = dataframe.question1
        self.question2 = dataframe.question2
        self.targets = dataframe.is_duplicate
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def tokenize(self, question1, question2):
        question1 = " ".join(question1.split())
        question2 = " ".join(question2.split())

        inputs = self.tokenizer.encode_plus(
            question1,
            question2,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        # token_type_ids = inputs["token_type_ids"]
        return ids, mask  # ,token_type_ids

    def __getitem__(self, index):
        ids, mask = self.tokenize(str(self.question1[index]),
                                  str(self.question2[index]))
        # ids2,mask2,token_type_ids2 = self.tokenize(str(self.question1[index]))

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class RankModel(nn.Module):
    def __init__(self, num_labels=2):
        super(RankModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


model = RankModel()

if args.drop:
    layer_list = model.bert.encoder.layer

    remove_layers = args.remove_layers
    if remove_layers is not "":
        layer_indexes = [int(x) for x in remove_layers.split(",")]
        layer_indexes.sort(reverse=True)
        for layer_idx in layer_indexes:
            if layer_idx < 0:
                print("Only positive indices allowed")
                sys.exit(1)
            del(layer_list[layer_idx])
            print("Removed Layer: ", layer_idx)

        model.bert.config.num_hidden_layers = len(layer_list)

device = 'cuda:0' if cuda.is_available() else 'cpu'

model.to(device)


class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0.4):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cos_sim = F.cosine_similarity(output1, output2)
        loss_cos_con = torch.mean((1 - label) * torch.div(torch.pow((1.0 - cos_sim), 2), 4) +
                                  (label) * torch.pow(cos_sim * torch.lt(cos_sim, self.margin), 2))
        return loss_cos_con


criterion = CosineContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_size = 0.8
train_dataset = df.sample(
    frac=train_size, random_state=200).reset_index(drop=True)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = SiameseNetworkDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = SiameseNetworkDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 0
               }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


def train(epoch):
    model.train()
    progress_bar = tqdm(training_loader)
    for _, data in enumerate(progress_bar):
        ids, mask = data['ids'], data['mask']
        targets = data['targets'].to(device, dtype=torch.long)
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        output = model(ids, mask)
        loss = F.cross_entropy(output, targets)
        if _ % 500 == 0:
            print(f'Step: {_}, Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for epoch in range(EPOCHS):
    train(epoch)
