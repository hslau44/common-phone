import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from segmentation import PhonemeSegmentor


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
        length = list(inputs.keys())[0]
        rows = []
        for i in range(length):
            row = {}
            for key in inputs.keys():
                row[key] = outputs[key][i]
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
          "./",
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