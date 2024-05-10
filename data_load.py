import pandas as pd
import torch
import random
import math
import os
import numpy as np
from torch.utils.data import Dataset

class BugDataset(Dataset):
    """Dataset for sequence-to-sequence generative models, i.e., BART"""

    def __init__(self, data_dir, dataset, tokenizer, text_max_token_len: int=512,
            summary_max_token_len: int=128):

        self.data_dir = data_dir
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data = None
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

        self.data = pd.read_csv(os.path.join(data_dir, dataset))
        self.data.dropna(inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]

        text_encoding = self.tokenizer(
            data_row['text'],
            max_length=self.text_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        summary_encoding = self.tokenizer(
            data_row['summary'],
            max_length=self.summary_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = summary_encoding['input_ids']
        labels[labels == 0] = -100

        return dict(
            text=data_row["text"],
            summary=data_row["summary"],
            text_input_ids=text_encoding['input_ids'].flatten(),
            text_attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding['attention_mask'].flatten()
        )


class StackTraceExtract:
    def __int__(self):
        pass
    def extract(self, text: str):
        res = set()
        search1 = re.compile("\w+\(([\w]+)\.\w+:\d+\)")
        find_result = search1.findall(text)
        search2 = re.compile("[A-Z][a-z]+")
        if len(find_result) != 0:
            for i in find_result:
                find_result2 = search2.findall(i)
                for j in find_result2:
                    res.add(j)
        return list(res)

class ApacheInfoExtract:
    def __int__(self):
        pass
    def extract(self, text: str):
        res = set()
        search1 = re.compile("/(\w+)\.\w+:\d+")
        find_result = search1.findall(text)
        # search2 = re.compile("[A-Z][a-z]+")
        if len(find_result) != 0:
            for i in find_result:
                res.add(i)
        return list(res)
        
class RefineHTML:
    def __int__(self):
        pass
    def extract(self, text: str):
        # \w + &  # \d+;\w*
        search = re.compile("#?&amp;|&?gt;|&?lt;|&#\d+;", flags=re.IGNORECASE)
        text1 = re.sub(search, " ", text)
        # search2 = re.compile("[A-Z][a-z]+")
        return text1