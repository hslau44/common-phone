import os
import json
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
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
from utils import write_json


def get_metadata(path,_set=None,_locale=None):
    metadata = pd.read_csv(path)
    if _set is not None:
        metadata = metadata[(metadata['set'] == _set)]
        if isinstance(_set,list):
            metadata = metadata[(metadata['set'].isin(_set))]
        else:
            metadata = metadata[(metadata['set'] == _set)]
    if _locale is not None:
        if isinstance(_locale,list):
            metadata = metadata[(metadata['set'].isin(_locale))]
        else:
            metadata = metadata[(metadata['locale'] == _locale)]
    return metadata


def get_vocab_dict(data,pad=None,unk=None):
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


# model
class ConvProjection(nn.Module):
    
    def __init__(self,in_features,out_features):
        super(ConvProjection, self).__init__()
        kernel_size=2 # 4
        stride=1 # 2
        self.layer = nn.ConvTranspose1d(in_features,out_features,kernel_size=kernel_size,stride=stride,padding=0)
    
    def forward(self,x):
        x = x.permute(0,2,1)
        return self.layer(x).permute(0,2,1)
    
    
class CustomWav2Vec2Segmentation(nn.Module):
    
    def __init__(self,model_checkpoint,num_labels,sr=16000):
        super(CustomWav2Vec2Segmentation, self).__init__()
        self.sr = sr
        self.model = Wav2Vec2ForAudioFrameClassification.from_pretrained(model_checkpoint,num_labels=num_labels)
        self.model.classifier = ConvProjection(self.model.classifier.in_features,num_labels)
        
    
    def forward(self,input_values,labels):
        x = input_values
        bs = x.shape[0]
        x = x.view(-1,self.sr)
        x = self.model(x)
        x.logits = x.logits.reshape(bs,-1,x.logits.shape[-1])
        return x


# loss    
def nll_loss(logits,labels):
    logits = logits.reshape(-1,logits.shape[-1])
    labels = labels.flatten()
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
    return loss


# metric
def avg_sample_acc(predictions, references):
    assert len(predictions) == len(references), f"length not equal: {len(predictions)} != {len(references)}"
    assert len(predictions.shape) == 3 and len(references.shape) == 2
    assert predictions.shape[1] == references.shape[1]
    predictions = np.argmax(predictions,axis=-1) 
    sample_accs = [metrics.accuracy_score(ref,pred) for ref,pred in zip(references,predictions)]
    avg = np.mean(sample_accs)
    return {'avg_sample_acc': avg}
    

def compute_avg_sample_acc(pred):
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


if __name__ == "__main__":
    
    model_checkpoint = "speech31/wav2vec2-large-english-phoneme-v2"
    PATH = 'data/metadata.csv'
    repo_name = 'hslau44/common-phone-dev'
    output_dir = 'outputs/exp_01'

    seed = 42
    num_train_epochs = 1
    batch_size = 8
    per_device_train_batch_size = batch_size # maximum 16 for g4dn.xlarge
    per_device_eval_batch_size = batch_size
    evaluation_strategy = 'epoch'
    save_strategy='epoch'
    logging_strategy='steps'
    action_step = 10
    save_steps = action_step
    eval_steps = action_step
    logging_steps = action_step
    learning_rate = 0.0003
    adam_beta1,adam_beta2 = 0.9,0.999
    adam_epsilon = 1e-08
    weight_decay = 0.005
    lr_scheduler_type ='linear'
    warmup_steps = 0
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_mdl_dir = os.path.join(output_dir,'models')
    output_log_dir = os.path.join(output_dir,'logs')
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

    train_metadata = get_metadata(PATH,'train','en')#.iloc[:12*batch_size,:]
    valid_metadata = get_metadata(PATH,'dev','en')#.iloc[:2*batch_size,:]
    
    trainset  = PhonemeDetailsDataset(train_metadata)
    validaset = PhonemeDetailsDataset(valid_metadata)
    
    resolution = 0.02
    manual_t_end = 8
    segmentor = PhonemeSegmentor(tokenizer=tokenizer,resolution=resolution,pad_token=pad_token)
    data_collator = TrainingDataProcessor(
        sampling_rate=16000,
        resolution=resolution,
        t_end=manual_t_end,
        pad_to_sec=1.0,
        tokenizer=segmentor
    )
    
    model = CustomWav2Vec2Segmentation(model_checkpoint=model_checkpoint,num_labels=tokenizer.vocab_size)
    
    training_config = {}
    
    group_by_length = False 
    remove_unused_columns = False
    
    training_config.update(
        output_dir=output_dir,
        group_by_length=group_by_length,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        seed=seed,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        num_train_epochs=num_train_epochs,
        fp16=torch.cuda.is_available(),
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        save_total_limit=2,
        push_to_hub=False,
        logging_dir=output_log_dir,
        remove_unused_columns=remove_unused_columns
    )
    
    training_config.update(
        optim="adafactor",
        gradient_checkpointing=False,
        tf32=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
    )
    

    training_args = TrainingArguments(**training_config)
    
    print("Training runs on: ",training_args.device)

    trainer = CustomTrainer(
        model=model,
        data_collator=data_collator, # data_collator,
        args=training_args,
        compute_metrics=compute_avg_sample_acc,
        train_dataset=trainset,
        eval_dataset=validaset,
#         tokenizer=segmentor,
    )

    
#     test_dataflow(model,trainset,data_collator)
    
    trainer.train()
    
    eval_result = trainer.evaluate(eval_dataset=validaset)

    print(f"***** Eval results *****")
    write_json(eval_result,os.path.join(output_log_dir,'eval_result.json'))

    # Saves the model to s3
    trainer.save_model(output_mdl_dir)
    
    print("***** Completed *****")




