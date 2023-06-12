from abc import ABC
import json
import logging
import os
import importlib
import torch
from ts.torch_handler.base_handler import BaseHandler
from . import models
from .models import CustomWav2Vec2Segmentation
from .data_utils import Audioloader, AudioArrayProcessor, PhonemeDetailProcessor


class ModelHandler(ABC):
    
    def __init__(self, config):
        self.config = config
        
    def __call__(self,data):
        raise NotImplementedError()
        
    def preprocess(self, data):
        raise NotImplementedError()
        
    def inference(self, inputs):
        raise NotImplementedError()

    def postprocess(self, data):
        raise NotImplementedError()
        

class PhonemeSegModelHandler(ModelHandler):
    
    def __init__(self, config):
        
        device = 'cpu'
        if config.get('gpu_num'):
            device = f"cuda:{gpu_num}"
        elif config.get('device'):
            device = "cuda"
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.data_loader = Audioloader(**config)
        self.data_worker = AudioArrayProcessor(**config)
        self.label_worker = PhonemeDetailProcessor(**config)
        self.model = self._load_model(config)
        self.model.to(self.device)
        self.model.eval()
        super().__init__(config=config)
        
        
    def __call__(self,data):
        
        inputs = self.preprocess(data)
        
        outputs = self.inference(inputs)
        
        result = self.postprocess(outputs)
        
        return result
        
    def preprocess(self, data):
        
        audio_arrays = self.data_loader(data)

        inputs = self.data_worker(audio_arrays)
        
        inputs['input_values'] = inputs['input_values'].to(self.device)
        
        return inputs
    
    def inference(self, inputs):
        
        with torch.no_grad():
            pred = self.model(inputs)
            
        return pred

    def postprocess(self, inference_output):

        inference_output['logits'] = inference_output['logits'].to('cpu')
        
        output = self.label_worker.inverse_transform(inference_output)
        
        return output
    
    def _load_model(self,config):
        
        model_cls = getattr(models,config['model_cls'])
        
        model = model_cls(**config)
        
        return model 
    
    
class CustomWav2Vec2SegmentationHandler(PhonemeSegModelHandler):

    def __init__(self,config):
        super().__init__(config=config)
        
    def _load_model(self,config):
        model = CustomWav2Vec2Segmentation(**config)
        model_fp = config['model_dir']
        load_device = torch.device('cpu')
        state_dict = torch.load(model_fp,map_location=load_device)
        model.load_state_dict(state_dict, strict=True)
        return model
    
    def inference(self, inputs):
        if inputs.get('labels'):
            pred = self.model(**inputs)
        else:
            pred = self.model(**inputs,labels=None)
        return pred
