from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import count_letters


class CharTokenizedDataset(Dataset):
    def __init__(self, sentences: List[str], vocab: List[str]):
        super().__init__()
        self.vocab = {s: i for i, s in enumerate(vocab)}
        self.sentences = sentences
        # Dataset is small enough that we can precompute all results.
        idxs, cnts = [], []
        for s in self.sentences:
            idx, cnt = self.process_sentence(s)
            idxs.append(idx)
            cnts.append(cnt)

        self.encoded_idxs = torch.LongTensor(idxs)
        self.encoded_counts = torch.LongTensor(np.array(cnts))

    def process_sentence(self, sentence: str):
        encoded_idx = [self.vocab[c] for c in sentence]
        encoded_count = count_letters(sentence)
        return encoded_idx, encoded_count

    @staticmethod
    def encode_sentence(sentence, vocab):
        vocab = {s: i for i, s in enumerate(vocab)}
        encoded_idx = [vocab[c.lower()] for c in sentence if c.lower() in vocab]
        encoded_count = count_letters(sentence)
        return encoded_idx, encoded_count

    def __getitem__(self, index):
        return self.encoded_idxs[index], self.encoded_counts[index]

    def __len__(self):
        return len(self.sentences)

    def get_dataloader(
        self, batch_size, num_workers=0, pin_memory=True, shuffle=True
    ) -> DataLoader:
        return DataLoader(
            self,
            # sampler=torch.utils.data.RandomSampler(
            #     self, replacement=True, num_samples=int(1e10)
            # ),
            shuffle=shuffle,
            pin_memory=pin_memory,
            batch_size=batch_size,
            num_workers=num_workers,
        )
