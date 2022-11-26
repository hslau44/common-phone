import os
import numpy as np
import torch 
from tqdm import tqdm
from torch.utils.data import Dataset


def get_vocab_dict(data,pad=None):
    pad = 'PAD' if pad is None else pad
    dictionary = {pad:0}
    lb = 1
    for phonemes_detal in tqdm(data):
        phonemes_detal = eval(phonemes_detal)
        phonemes = phonemes_detal['label']
        for phoneme in phonemes:
            if phoneme not in dictionary.keys():
                dictionary[phoneme] = lb
                lb += 1
    dictionary['UNK'] = lb
    return dictionary


def get_phonemes_segment(phonemes_detal,mapping,t_end,res):
    arr = [0]*res
    phonemes_detal = eval(row['phonetic_detail']) if type(row['phonetic_detail']) is not dict else row['phonetic_detail']
    for a,b,label in zip(phonemes_detal['start'],phonemes_detal['end'],phonemes_detal['label']):
        a = int(np.round(a/(t_end/res)))
        b = int(np.round(b/(t_end/res)))
        arr[a:b] = [mapping[label]]*len(arr[a:b])
    return arr


def char_processing(chars):
    return "".join([c if c != '(...)' else ' ' for c in chars])


def get_phonemes_string(phonemes_detal):
    phonemes_detal = eval(phonemes_detal) if type(phonemes_detal) is not dict else phonemes_detal
    processed_string = char_processing(phonemes_detal['label'])
    return processed_string


class PhonemeSegmentationDataset(Dataset):
    
    def __init__(self,metadata,pipeline,tokenizer):
        self.metadata = metadata
        self.pipeline = pipeline
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(metadata)
    
    def __getitem__(self,idx):
        example = {}
        row = metadata.iloc[idx,:]
        fpath = row['file_name']
        phonetic_detail = row['phonetic_detail']
        # phonetic_detail = eval(row['phonetic_detail']) if type(row['phonetic_detail']) is not dict else row['phonetic_detail']
        example.update(self.pipeline(fpath))
        example.update(self.tokenizer(phonetic_detail))
        return example 