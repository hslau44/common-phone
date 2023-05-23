from abc import ABC
import json
import logging
import os
import importlib
import librosa
import torch
from ts.torch_handler.base_handler import BaseHandler
from . import models
from .models import CustomWav2Vec2Segmentation
from .data_utils import AudioArrayProcessor, PhonemeDetailProcessor


logger = logging.getLogger(__name__)


# class PhonemeSegModelHandler(BaseHandler, ABC):
    
#     def __init__(self,config,model_cls,save_path):
#         super(PhonemeSegModelHandler, self).__init__()
#         self.data_worker = AudioArrayProcessor(**config)
#         self.label_worker = PhonemeDetailProcessor(**config)
#         self.model = self.loaded_model(config,model_cls,save_path)
#         pass
    
#     def __call__(self,inputs):
#         inputs = self.data_worker(inputs)
#         inputs = self.forward(inputs)
#         inputs = self.label_worker.inverse_transform(inputs)
#         return inputs
    
#     @torch.no_grad()
#     def forward(self,inputs):
#         return self.model(inputs)
    
#     def loaded_model(self,config,model_cls,save_path):
#         model = model_cls(**config)
#         return model

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
  
    
class PhonemeSegModelHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a audio array (string) and
    as input and returns the phoneme detail based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        
        self.manifest = ctx.manifest
        config = ctx.system_properties
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_worker = AudioArrayProcessor(**config)
        self.label_worker = PhonemeDetailProcessor(**config)
        self.model = self._load_model(config)
        self.model.to(self.device)
        self.model.eval()
        
        self.initialized = True
    
    def _load_model(self,config):
        
        model_cls = getattr(models,config['model_cls'])
        model = model_cls(**config)
        
        logger.debug(f'Model loaded successfully: model class: {repr(model_cls)}')
        return model 

    def preprocess(self, data):
            
        logger.info(f"Received number of filepaths: {len(data)}")
        
        audio_arrays = [{'input_values':sf.read(fpath)} for fpath in data]

        inputs = self.data_worker(audio_arrays)
        
        return inputs

    def inference(self, inputs):
        # NOTE: This makes the assumption that your model expects text to be tokenized  
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit 
        # its expected input format.
        pred = self.model(inputs)
        return pred

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        output = self.label_worker.inverse_transform(inference_output)
        return output

    
class CustomWav2Vec2SegmentationHandler(PhonemeSegModelHandler):

    def __init__(self):
        super().__init__()
        
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


def handle(data, context):
    _service = CustomWav2Vec2SegmentationHandler()
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
