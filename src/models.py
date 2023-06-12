import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForAudioFrameClassification


ACTIVATIONS = {
    'gelu':nn.GELU,
    'relu':nn.ReLU,
    'none':nn.Identity,
    'None':nn.Identity,
}


def truncate_arr_from_centre(arr,keep=None):
    a,b = [],[]
    i = -1
    for i in range(len(arr)//2):
        a.append(arr[i])
        b.append(arr[-i-1])
    if len(arr) % 2 == 1:
        a.append(arr[i+1])
    if keep is not None:
        if keep % 2 == 1:
            a = a[:keep//2 + 1]
        else:
            a = a[:keep//2]
        b = b[:keep//2]
    return a + b[::-1]


def layer_truncation(model,keep_num_layers,verbose=1):
    length = len(model.layers)
    assert keep_num_layers > 0
    assert length >= keep_num_layers
    idxs = truncate_arr_from_centre(range(length),keep=keep_num_layers)
    print(f"Keep layer: {idxs}") if verbose else None
    model.layers = nn.ModuleList([model.layers[i] for i in idxs])

    
class ConvProjectionLayer(nn.Module):
    
    def __init__(self,in_features,out_features,kernel_size=1,stride=1,padding=0,actv='gelu'):
        super(ConvProjectionLayer, self).__init__()
        self.conv = nn.ConvTranspose1d(in_features,out_features,kernel_size=kernel_size,stride=stride,padding=padding)
        self.activation = ACTIVATIONS[actv]() if actv is not None else nn.Identity()
        
    def forward(self,x):
        return self.activation(self.conv(x))
    

class ConvProjection(nn.Module):
    
    def __init__(self,in_features,out_features,num_layers=1,hid_actv='gelu',last_actv=None,resolution=0.02):
        super(ConvProjection, self).__init__()
        res_config = {0.02:(2,1),0.01:(4,2),0.005:(8,4)}
        kernel_size,stride = res_config[resolution]
        self.layers = nn.ModuleList()
        for i in range(num_layers-1):
            self.layers.append(ConvProjectionLayer(in_features,in_features,kernel_size=1,stride=1,padding=0,actv=hid_actv))
        self.layers.append(ConvProjectionLayer(in_features,out_features,kernel_size=kernel_size,stride=stride,padding=0,actv=None))
        self.activation = ACTIVATIONS[last_actv]() if last_actv is not None else nn.Identity()
    
    def forward(self,x):
        x = x.permute(0,2,1)
        for layer in self.layers:
            x = layer(x)
        return self.activation(x.permute(0,2,1))
    

class CustomWav2Vec2Segmentation(nn.Module):
    
    def __init__(self,
                 model_checkpoint,
                 num_labels,
                 num_encoders=None,
                 num_convprojs=1,
                 conv_hid_actv='gelu',
                 conv_last_actv=None,
                 sr=16000,
                 resolution=0.02,
                 freeze_encoder=False,
                 **kwargs,
                ):
        super(CustomWav2Vec2Segmentation, self).__init__()
        self.sr = sr
        self.model = Wav2Vec2ForAudioFrameClassification.from_pretrained(model_checkpoint,num_labels=num_labels)
        if num_encoders is not None:
            layer_truncation(
                model=self.model.wav2vec2.encoder,
                keep_num_layers=num_encoders,
                verbose=0,
            )
        self.model.classifier = ConvProjection(
            in_features=self.model.classifier.in_features,
            out_features=num_labels,
            num_layers=num_convprojs,
            hid_actv=conv_hid_actv,
            last_actv=conv_last_actv,
            resolution=resolution
        )
        if freeze_encoder:
            self.change_grad_state('encoder',range(30),False)
    
    def forward(self,input_values,labels):
        x = input_values
        bs = x.shape[0]
        x = x.view(-1,self.sr)
        x = self.model(x)
        x.logits = x.logits.reshape(bs,-1,x.logits.shape[-1])
        return x
    
    def change_grad_state(self,section:str,layer_nums:list,state:bool):
        if section == 'encoder':
            for i,layer in enumerate(self.model.wav2vec2.encoder.layers):
                if i in layer_nums:
                    for param in layer.parameters():
                        param.requires_grad = state
        elif section == 'convproj':
            for i,layer in enumerate(self.model.classifier.layers):
                if i in layer_nums:
                    for param in layer.parameters():
                        param.requires_grad = state


class BaseModel(nn.Module):
    
    def __init__(self,config):
        self.config = config
        super().__init__()
    
    def forward(self,inputs:list) -> list:
        raise NotImplementedError()