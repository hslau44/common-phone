import argparse
import json

ARGS_FILEPATHS = [
    'training_config.json',
    'segmentation_config.json',
]


def write_json(items,fp):
    with open(fp, 'w') as file:
        json.dump(items, file)
        
def read_json(fp):
    with open(fp,'r') as file:
        data = json.load(file)
    return data


def get_default_arg(filepaths=ARGS_FILEPATHS):
    args = {}
    for fp in filepaths:
        args.update(read_json(fp))
    return args

def extract_args_by_default_config(args,default_config):
    config = {}
    for k,v in vars(args).items():
        if k in default_config.keys():
            config[k] = v
    return config


def add_argument_from_default_config(parser,default_config, add_as_default=None):
    for k,v in default_config.items():
        default_value = None
        if add_as_default:
            default_value = v
        if isinstance(v,list):
            parser.add_argument(f"--{k}", nargs='+', type=type(v[0]), default=default_value)
        else:
            parser.add_argument(f"--{k}", type=type(v), default=default_value)
        

def set_argparser_by_default_configs(*config_files):
    configs = [read_json(config_file) for config_file in config_files]
    parser = argparse.ArgumentParser()
    for config in configs:
        add_argument_from_default_config(parser,config)
    return parser
