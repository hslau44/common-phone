import numpy as np
import torch

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
