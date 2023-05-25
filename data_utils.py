import os
import soundfile as sf
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer
from .config import ROOT_DIR


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


class Audioloader:
    
    def __init__(self,sampling_rate,**kwargs):
        self.sampling_rate = sampling_rate
        self.fpath_key = 'file_name'
        self.arr_key = 'input_values'
    
    def __call__(self,data):
        if isinstance(data[0],dict):
            data = self._process_dics(data)
        elif isinstance(data[0],str):
            data = self._process_fpaths(data)
        else:
            raise ValueError("incorrect input format")
        return data 
    
    def _process_dics(self,dics):
        for dic in dics:
            fpath = dic[self.fpath_key]
            arr = self._load_audio(fpath)
            dic[self.arr_key] = arr
        return dics
    
    def _process_fpaths(self,fpaths):
        data = []
        for fpath in fpaths:
            arr = self._load_audio(fpath)
            dic = {self.arr_key:arr}
            data.append(dic)
        return data
    
    def _load_audio(self,fpath):
        arr, _ = librosa.load(fpath,sr=self.sampling_rate)
        return arr


class PhonemeSegmentor:
    """


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


# loss    
def nll_loss(logits,labels):
    logits = logits.reshape(-1,logits.shape[-1])
    labels = labels.flatten()
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
    return loss


# metric
def avg_sample_acc(predictions, references):
    """calculate the accuracy of each sample in a batch and average the score"""
    if len(predictions) != len(references):
        raise ValueError(f"length not equal: {len(predictions)} != {len(references)}")
    if predictions.shape[1] != references.shape[1]:
        raise ValueError(f"Time interval not equal")
    if len(predictions.shape) != 3 or len(references.shape) != 2:
        raise ValueError("Dim not correct")
    predictions = np.argmax(predictions,axis=-1) 
    sample_accs = [metrics.accuracy_score(ref,pred) for ref,pred in zip(references,predictions)]
    avg = np.mean(sample_accs)
    return {'avg_sample_acc': avg}
    

def compute_avg_sample_acc(pred):
    """wrapping function to feed HF style prediction into metric """
    p = pred.predictions
    r = pred.label_ids
    return avg_sample_acc(p, r)


class BaseProcessor(object):
    
    def __init__(self,**kwargs):
        pass
    
    def __call__(self,inputs):
        raise NotImplementedError()


class BatchProcessor(BaseProcessor):
    
    def __init__(self, to_type=None, from_type=None, **kwargs):
        self.to_type = to_type
        self.from_type = from_type
        super().__init__()
        
    def to_batch(self,inputs):
        if self.to_type == 'pt':
            inputs = np.concatenate(inputs)
            inputs = torch.Tensor(inputs)
        else:
            raise ValueError(f"Currently not support type: '{self.to_type}'")
        return inputs
    
    def from_batch(self,inputs):
        length = len(inputs[list(inputs.keys())[0]])
        rows = []
        for i in range(length):
            row = {}
            for key in inputs.keys():
                row[key] = inputs[key][i]
            rows.append(row)
        return rows


class AudioArrayProcessor(BatchProcessor):
    
    def __init__(self,t_end,sampling_rate,rtn_type='pt',**kwargs):
        self.t_end = t_end
        self.sampling_rate = sampling_rate
        self.rtn_type = rtn_type
        super().__init__(to_type=rtn_type)
        
    def __call__(self,inputs):
        return self.transform(inputs)

    def transform(self,inputs):
        audio_inputs = []
        for example in inputs:
            x = example["input_values"]
            x = self.process_values(x)
            audio_inputs.append(x)
        batch = {"input_values":self.to_batch(audio_inputs)}
        return batch
    
    def process_values(self,x):
        target_len = int(self.sampling_rate*self.t_end)
        pad_size = int(target_len - x.shape[0])
        if pad_size > 0:
            x = np.pad(x,(0,pad_size))
        elif pad_size < 0:
            x = x[:target_len]
        x = x.reshape(1,-1)
        return x
    

class PhonemeDetailProcessor(BatchProcessor):

    def __init__(self,model_checkpoint,resolution,t_end,rtn_type='pt',**kwargs):
        hf_config = AutoConfig.from_pretrained(model_checkpoint)
        tokenizer_type = hf_config.model_type if hf_config.tokenizer_class is None else None
        hf_config = hf_config if hf_config.tokenizer_class is not None else None

        pad_token="(...)"
        unk_token="UNK"
        tokenizer = AutoTokenizer.from_pretrained(
          ROOT_DIR, # "./",
          config=hf_config,
          tokenizer_type=tokenizer_type,
          unk_token=unk_token,
          pad_token=pad_token,
        )
        self.segmentor = PhonemeSegmentor(tokenizer=tokenizer,resolution=resolution,t_end=t_end,pad_token=pad_token)
        super().__init__(to_type=rtn_type)
        
    def __call__(self,inputs):
        return self.transform(inputs)

    def transform(self,inputs):
        labels = []
        for example in inputs:
            y = example["labels"]
            y = self.process_labels(y)
            labels.append(y)
        batch = {"labels":self.to_batch(labels)}
        return batch
    
    def process_labels(self,y):
        y = self.segmentor.encode(y)
        y = np.reshape(y,(1,-1))
        return y
    
    def to_batch(self,inputs):
        return super().to_batch(inputs).long()
    
    def inverse_transform(self,inputs):
        outputs = self.from_batch(inputs)
        for i in range(len(outputs)):
            row = outputs[i]
            row = self.process_row(row)
            outputs[i] = row
        return outputs

    def process_row(self,row):
        logits = row['logits']
        row['labels'] = self.process_logits(logits)
        row.pop('logits')
        return row

    def process_logits(self,logits):
        arr = logits.detach().numpy()
        arr = np.argmax(arr,-1)
        labels = self.segmentor.decode(arr)
        return labels


class TrainingDataProcessor(BaseProcessor):
    """
    Object that collates samples generated by 
    PhonemeDetailsDataset as batch with desired format for 
    huggingface transformers models to read.  
    Can use as collate_fn for Pytorch Dataloader arguments
    
    Attributes
    ---
        sampling_rate: int
            sampeling rate of the audio 
        resolution: float
            resolution of label segment in second 
        tokenizer: PhonemeSegmentor
            tokenizer to translate phoneme_detail to array 
            with size (batch,t,num_label), where t is 
            t_end/resolution
        t_end: int
            length of array in sencod 
        pad_to_sec: int
            nearest second the array pad to 
        rtn_type: str
            tensor type to be return, currently only 
            support pytorch 'pt' 
            
    """
    def __init__(self,model_checkpoint,sampling_rate,resolution,t_end,rtn_type='pt',**kwargs):
        self.arr_processor = AudioArrayProcessor(
            t_end=t_end,
            sampling_rate=sampling_rate,
            rtn_type=rtn_type
        )
        self.label_processor = PhonemeDetailProcessor(
            model_checkpoint=model_checkpoint,
            resolution=resolution,
            t_end=t_end,
            rtn_type=rtn_type
        )

    def transform(self,inputs):
        batch = {}
        batch.update(self.arr_processor.transform(inputs))
        batch.update(self.label_processor.transform(inputs))
        return batch

    def __call__(self,inputs):
        return self.transform(inputs)


# Data generator
class PhonemeDetailsDataset(Dataset):
    """
    Pytorch Dataset read the row of the metadata and returns 
    the follows items as dict:
        input_values: np.ndarray
            the audio array loaded by reading the column 
            'file_name' of a row in the metadata
        labels: str
            phonetic segmentation in string extract directly
            from column 'phonetic_detail' of a row in the
            metadata
    
    Attributes
    ---
        metadata: pd.DataFrame
            a pandas dataframe object that contains the 
            informations about the audio 
        data_dir: str
            the directory of all data, same as the directory 
            where metadata.csv is located, default 'data'
    
    """
    def __init__(self,_set,mode,**kwargs):
        data_dir = kwargs['datadir'] # <------- find correct arg name
        path = os.path.join(data_dir,'metadata.csv')
        if _set == 'test':
            _locale = kwargs['test_locales']
        else:
            _locale = kwargs['train_locales']
        metadata = get_metadata(path=path,_set=_set,_locale=_locale)
        if mode == 'debug':
            metadata = metadata.copy().iloc[:64,:]
            metadata.reset_index(drop=True, inplace=True)
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


class SegmentationDataLoader(DataLoader):

    def __init__(self,data_dir,model_checkpoint,sampling_rate,resolution,t_end,_set,**kwargs):
        metadata_fp = os.path.join(data_dir,"metadata.csv")
        _locale = None
        if _set == 'test':
            _locale = kwargs.get('test_locales')
        else:
            _locale = kwargs.get('train_locales')
        metadata = get_metadata(path=metadata_fp,_locale=_locale,_set=_set)
        dataset = PhonemeDetailsDataset(metadata,data_dir=data_dir)
        collate_fn = TrainingDataProcessor(
            model_checkpoint=model_checkpoint,
            sampling_rate=sampling_rate,
            resolution=resolution,
            t_end=t_end,
            rtn_type='pt',
        )
        batch_size = kwargs.get('batch_size',1)
        shuffle = kwargs.get('shuffle',None)
        num_workers = kwargs.get('num_workers',0)
        super().__init__(dataset=dataset,collate_fn=collate_fn,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
