import os
import argparse
import json
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoFeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForAudioFrameClassification,
    Trainer, 
    TrainingArguments,
)
from datasets import load_dataset
import evaluate
from sklearn import metrics
from utils import *
from models import CustomWav2Vec2Segmentation


def get_metadata(path,_set=None,_locale=None):
    """
    Return metadata with specific set and locale
    
    Args
    ----
        path : str
            filepath of metadata.csv
        _set : str/list[str]/bool
            the set(s) it contains, only support combination 
            of ['train','dev','test'], default None to select 
            all sets
        _locale:
            the locale(s) it contain, default None to select 
            all sets
    
    Return
    ------
        metadata : pd.DataFrame
            selected metadata 
    """
    metadata = pd.read_csv(path)
    if _set is not None:
        if isinstance(_set,list):
            metadata = metadata[(metadata['set'].isin(_set))]
        else:
            metadata = metadata[(metadata['set'] == _set)]
    if _locale is not None:
        if isinstance(_locale,list):
            metadata = metadata[(metadata['locale'].isin(_locale))]
        else:
            metadata = metadata[(metadata['locale'] == _locale)]
    return metadata


def get_vocab_dict(data,pad=None,unk=None):
    """
    Return dictionary of token from a sequence of 
    phonemes_detail in the metadata
    
    Args
    ----
        data : list[str]
            sequence of phonemes_detail, this method will transform
            the string in phonemes_detail as dict 
        pad : str
            padding tokken, default : 'PAD'
        unk: str
            unknown tokken, default : 'UNK'
    
    Return
    ------
        vocab_dict : dict
            dictionary of token
            
    Example
    -------
    
        pad = '[PAD]'
        unk = '[UNK]'
        data = metadata['phonemes_detail']
        vocab_dict = get_vocab_dict(data,pad,unk)
        
    """
    pad = 'PAD' if pad is None else pad
    unk = 'UNK' if unk is None else unk
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


class PhonemeSegmentor:
    """
    Return dictionary of token from single 
    phonemes_detail in the metadata
    
    Attributes
    ----
        data : list[str]
            sequence of phonemes_detail, this method will 
            transform
            the string in phonemes_detail as dict 
        pad : str
            padding tokken, default : 'PAD'
        unk: str
            unknown tokken, default : 'UNK'
    
    Methods
    ------
        encode(phonemes_detal,**kwargs)
            tokenize and place the token into segment of array 
            accordingly
        
        decode(arr)
            decode the array and return a dict in the form of 
            phonemes_detail

    """
    def __init__(self,tokenizer,resolution,t_end=None,pad_token="[PAD]"):
        self.tokenizer = tokenizer
        self.t_end = t_end
        self.resolution = resolution
        self.pad_token = pad_token
        pass
    
    def encode(self,phonemes_detal,**kwargs):
        t_end = kwargs.get('t_end',self.t_end) if kwargs.get('t_end',self.t_end) else self.get_t_end(phonemes_detal)
        arr = [self.tokenizer.encode(self.pad_token)[0] for _ in range(int(np.round(t_end/self.resolution)))]
        phonemes_detal = eval(phonemes_detal) if type(phonemes_detal) is not dict else phonemes_detal
        for a,b,label in zip(phonemes_detal['start'],phonemes_detal['end'],phonemes_detal['label']):
            a = int(np.round(a/self.resolution))
            b = int(np.round(b/self.resolution))
            arr[a:b] = [self.tokenizer.encode(label)[0]]*len(arr[a:b])
        return arr
    
    def decode(self,arr):
        dec = len(str(self.resolution).split(".")[1]) if not isinstance(self.resolution,int) else 0
        d = {k:[] for k in ['start','end','label']}
        a = 0*self.resolution
        if len(arr) == 1:
            i = 0
            d['start'].append(round(a*self.resolution,dec))
            d['end'].append(round((i+1)*self.resolution,dec))
            label = self.tokenizer.decode(arr[i])
            d['label'].append(label)
        else:
            for i in range(len(arr)-1):
                if arr[i+1] != arr[i]:  
                    d['start'].append(round(a*self.resolution,dec))
                    d['end'].append(round((i+1)*self.resolution,dec))
                    label = self.tokenizer.decode(arr[i])
                    d['label'].append(label)
                    a = i+1
            d['start'].append(round(a*self.resolution,dec))
            d['end'].append(round((i+2)*self.resolution,dec))
            label = self.tokenizer.decode(arr[i])
            d['label'].append(label)
        return d
    
    def get_t_end(self,phonemes_detal):
        phonemes_detal = eval(phonemes_detal) if type(phonemes_detal) is not dict else phonemes_detal
        return phonemes_detal['label'][-1]


# Data generator
class PhonemeDetailsDataset(Dataset):
    
    def __init__(self,metadata,data_dir='data'):
        self.metadata = metadata
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):
        row = self.metadata.iloc[idx,:]
        # audio input
        fpath = os.path.join(self.data_dir,row['file_name'])
        audio_input,sr = sf.read(fpath)
        # label
        label = row['phonetic_detail']
        # inputs
        example = {"input_values":audio_input,"labels":label}
        return example


# Data processor/collator    
class TrainingDataProcessor:
    
    def __init__(self,sampling_rate,resolution,tokenizer,t_end=None,pad_to_sec=1,rtn_type='pt'):
        self.sampling_rate = sampling_rate
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.t_end = t_end
        self.pad_to_sec = pad_to_sec
        self.rtn_type = rtn_type
 
    
    def __call__(self,inputs):
        audio_inputs,labels = [],[]
        t_end = self._get_max_t(inputs) if self.t_end is None else self.t_end
        for example in inputs:
            x = example["input_values"]
            x = self.process_values(x,t_end)
            y = example["labels"]
            y = self.process_labels(y,t_end)
            audio_inputs.append(x)
            labels.append(y)
        
        batch = self._rtn_batch(audio_inputs,labels)
        return batch
    
    def process_values(self,x,t_end):
        target_len = int(self.sampling_rate*t_end)
        pad_size = int(target_len - x.shape[0])
        if pad_size > 0:
            x = np.pad(x,(0,pad_size))
        elif pad_size < 0:
            x = x[:target_len]
        x = x.reshape(1,-1)
        return x
    
    def process_labels(self,y,t_end):
        # y = get_phonemes_segment(y,self.tokenizer,t_end,self.resolution)
        y = self.tokenizer.encode(y,t_end=t_end)
        y = np.reshape(y,(1,-1))
        return y
    
    def _get_max_t(self,inputs):
        # use 'input_values' max array length to calculate t 
        maxlen = int(np.max([item['input_values'].shape[0] for item in inputs]))
        max_t = maxlen/self.sampling_rate
        pad_res = self.pad_to_sec if self.pad_to_sec is not None else self.resolution
        max_t = np.ceil(max_t/pad_res)*pad_res
        return max_t 
    
    def _rtn_batch(self,audio_inputs,labels):
        if self.rtn_type == 'pt':
            audio_inputs = np.concatenate(audio_inputs)
            audio_inputs = torch.Tensor(audio_inputs)
            labels = np.concatenate(labels)
            labels = torch.Tensor(labels).long()
        else:
            raise ValueError(f"Currently not support type: '{self.rtn_type}'")
        return {"input_values":audio_inputs,"labels":labels}


# loss    
def nll_loss(logits,labels):
    logits = logits.reshape(-1,logits.shape[-1])
    labels = labels.flatten()
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
    return loss


# metric
def avg_sample_acc(predictions, references):
    """calculate the accuracy of each sample in a batch and average the score"""
    assert len(predictions) == len(references), f"length not equal: {len(predictions)} != {len(references)}"
    assert len(predictions.shape) == 3 and len(references.shape) == 2
    assert predictions.shape[1] == references.shape[1]
    predictions = np.argmax(predictions,axis=-1) 
    sample_accs = [metrics.accuracy_score(ref,pred) for ref,pred in zip(references,predictions)]
    avg = np.mean(sample_accs)
    return {'avg_sample_acc': avg}
    

def compute_avg_sample_acc(pred):
    """wrapping function to feed HF style prediction into metric """
    p = pred.predictions
    r = pred.label_ids
    return avg_sample_acc(p, r)


# trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = nll_loss(logits,labels)
        return (loss, outputs) if return_outputs else loss


def test_dataflow(model,dataset,data_collator,batch_size=2):
    
    dl = DataLoader(dataset=dataset,batch_size=batch_size,collate_fn=data_collator)
    example_inputs = next(iter(dl))
    input_values = example_inputs['input_values']
    labels = example_inputs['labels']
    
    if torch.cuda.is_available():
        model = model.cuda()
        input_values = input_values.cuda()
        labels = labels.cuda()

    print("example_inputs['input_values'].shape:   ",example_inputs['input_values'].shape)
    print("example_inputs['labels'].shape:   ",example_inputs['labels'].shape)

    with torch.no_grad():
        example_outputs = model(input_values,labels)

    logits = example_outputs.logits

    print("loss: ",nll_loss(logits,labels))
    
    if torch.cuda.is_available():
        model = model.cpu()
        logits = logits.cpu()
        labels = labels.cpu()
    
    print("metric: ",avg_sample_acc(logits,labels))
    return


def train(
    mode,
    model_checkpoint,
    train_locales,
    test_locales,
    sampling_rate,
    resolution,
    t_end,
    pad_to_sec,
    training_config,
    datadir,
    output_data_dir,
    num_encoders,
    num_convprojs,
    conv_hid_actv,
    conv_last_actv,
    **kwargs
    ):
    
    # compute config
    transformers.utils.logging.set_verbosity_error()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = torch.cuda.is_available()
    
    if model_checkpoint is None:
        model_checkpoint = "speech31/wav2vec2-large-english-phoneme-v2"
    print(f"Using checkpoint: {model_checkpoint}")

    # save_subfolder config
    output_dir = output_data_dir
    output_mdl_dir = os.path.join(output_dir,'models') 
    output_log_dir = os.path.join(output_dir,'logs') 

    # data config
    data_dir = datadir
    metadata_dir = os.path.join(data_dir,'metadata.csv')
    if not os.path.exists(metadata_dir):
        raise Exception(f'Metadata does not existed, datadir: {os.listdir(data_dir)}')
    
    train_metadata = get_metadata(metadata_dir,'train',train_locales) 
    valid_metadata = get_metadata(metadata_dir,'dev',test_locales) 
    trainset  = PhonemeDetailsDataset(train_metadata,data_dir) 
    validaset = PhonemeDetailsDataset(valid_metadata,data_dir) 
    print(f"language\n training:{train_locales}   length: {len(trainset)}\n test:{test_locales}   length: {len(validaset)}")
    
    # model and data-processor config 
    hf_config = AutoConfig.from_pretrained(model_checkpoint)
    tokenizer_type = hf_config.model_type if hf_config.tokenizer_class is None else None
    hf_config = hf_config if hf_config.tokenizer_class is not None else None
    
    pad_token="(...)"
    unk_token="UNK"
    tokenizer = AutoTokenizer.from_pretrained(
      "./",
      config=hf_config,
      tokenizer_type=tokenizer_type,
      unk_token=unk_token,
      pad_token=pad_token,
    )
    
    segmentor = PhonemeSegmentor(tokenizer=tokenizer,resolution=resolution,pad_token=pad_token)
    data_collator = TrainingDataProcessor(
        sampling_rate=sampling_rate,
        resolution=resolution, 
        t_end=t_end, 
        pad_to_sec=pad_to_sec, 
        tokenizer=segmentor 
    )

#     num_encoders = 3
#     num_convprojs = 2
#     conv_hid_actv = 'gelu'
#     conv_last_actv = None

    model = CustomWav2Vec2Segmentation(
        model_checkpoint,
        num_labels=tokenizer.vocab_size,
        num_encoders=num_encoders,
        num_convprojs=num_convprojs,
        conv_hid_actv=conv_hid_actv,
        conv_last_actv=conv_last_actv,
        resolution=resolution
    )
    
    if kwargs.get('freeze_encoder'):
        model.change_grad_state('encoder',range(30),False)
    
    training_config.update(
        output_dir=output_mdl_dir,
        logging_dir=output_log_dir,
        group_by_length = False, 
        remove_unused_columns = False,
        optim="adafactor",                        # <-------------------------------------------  check necessity 
        gradient_checkpointing=False,             # <-------------------------------------------  check necessity 
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
    )

    training_args = TrainingArguments(**training_config)

    trainer = CustomTrainer(
        model=model,
        data_collator=data_collator, 
        args=training_args,
        compute_metrics=compute_avg_sample_acc,
        train_dataset=trainset,
        eval_dataset=validaset,
    )
    
    print("Training runs on: ",training_args.device)

    if mode == "train":
        
        trainer.train()

        eval_result = trainer.evaluate(eval_dataset=validaset)

        print(f"***** Eval results *****")
        write_json(eval_result,os.path.join(output_log_dir,'eval_result.json')) 

        # Saves the model to s3
        trainer.save_model(output_mdl_dir) 
    
    else:
    
        test_dataflow(model,trainset,data_collator)
    
    print("***** Completed *****")
    
    return


if __name__ == "__main__":
    
    # arg 
    config_files = ['segmentation_config.json','training_config.json',]
    configs = [read_json(c) for c in config_files]
    
    parser = argparse.ArgumentParser()
    
    for config in configs:
        add_argument_from_default_config(parser,config)
    
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument("--output_data_dir", type=str, default='outputs')
    
    args, _ = parser.parse_known_args()
    
    args_dic = extract_args_by_default_config(args,configs[0])
    args_dic['training_config'] = extract_args_by_default_config(args,configs[1])
    args_dic['output_data_dir'] = args.output_data_dir
    args_dic['datadir'] = args.datadir
    
    print("ARGS:\n",args_dic,"\n")
    train(**args_dic)