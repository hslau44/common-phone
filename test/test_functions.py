from ..src.utils import get_default_arg
from ..src.hf_train import PhonemeDetailsDataset_ as PhonemeDetailsDataset

def test_loaded_model():
    res = 1
    
    args = get_default_arg(['src/segmentation_config.json'])
    args['num_convprojs'] = 2
    args['resolution'] = 0.02
    args['datadir'] = 'data'
    args['model_dir'] = 'outputs/checkpoints/2023-05-09_2-23/checkpoint-360/pytorch_model.bin'
    
    model = CustomWav2Vec2Segmentation(**args)
    
    # ['2023-05-09_0-23', '2023-05-09_1-31', '2023-05-09_0-50', '2023-05-09_1-57',\
    # '2023-05-09_1-0', '2023-05-09_2-23', '2023-05-09_1-16', '2023-05-09_0-39']
    
#     model_fp = args['model_dir']
#     state_dict = torch.load(model_fp,map_location=torch.device('cpu'))
#     model.load_state_dict(state_dict, strict=True)
    
#     if len(missed_key) > 0 or len(unexpected_key) > 0:
#         raise Exception("")

    
#     for key in state_dict.keys():
        
#         if key not in barebone_state_dict.keys():
        
#             shape = state_dict[key].shape
#             # shape = state_dict[key].shape
            
#             print(f">>>>>>>> {key} --- {shape} <<<<<<<<<<")
    
    # res = (missed_key, unexpected_key)
    
    res = model.__class__
    
    return res


def get_handle_args(args,dataset,num=5):
    
    class Context:
        
        def __init__(self,config):
            self.manifest = None
            self.system_properties = config
    
    cxt = Context(config=args)
    
    data = []
    for i in range(num):
        row = dataset[i]
        # row['data'] = row['input_values']
        row.pop('labels')
        # row.pop('input_values')
        data.append(row)
    
    return cxt, data


def test_serve_handle():
    
    res = 1
    
    args = get_default_arg(['src/segmentation_config.json'])
    args['num_convprojs'] = 2
    args['resolution'] = 0.02
    args['datadir'] = 'data'
    args['model_dir'] = 'outputs/checkpoints/2023-05-09_2-23/checkpoint-360/pytorch_model.bin'
    # args['model_cls'] = 'CustomWav2Vec2Segmentation'
    
    dataset = PhonemeDetailsDataset(_set='train',**args)
    
    cxt, data = get_handle_args(args=args,dataset=dataset,num=3)
    
    from src.serve import handle
    
        # if "input_values" not in list(data[0].keys()):
        #     raise KeyError("key 'input_values' does not exist")
    
    res = handle(data=data,context=cxt)
    return res

results = test_serve_handle()