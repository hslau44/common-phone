import os
import argparse
import json
import ast
from .config import ARGS_FILEPATHS

LITERAL_TO_BOOL = {'true': True, 'false': False, 'True': True, 'False': False}

# ARGS_FILEPATHS = [
#     'training_config.json',
#     'segmentation_config.json',
# ]


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


def add_argument_from_default_config(parser, default_config):
    for k in default_config.keys():
        parser.add_argument(f"--{k}", type=str)


def set_argparser_by_default_configs(*config_files):
    configs = [read_json(config_file) for config_file in config_files]
    parser = argparse.ArgumentParser()
    for config in configs:
        add_argument_from_default_config(parser,config)
    return parser


def process_namespace_arguments(namespace_args, default_config=None, add_default_value=False):
    args = {}
    for k, v in vars(namespace_args).items():
        if k in default_config.keys():
            default_value = default_config[k]
            if v is None:
                if add_default_value:
                    args[k] = default_value
                else:
                    continue
            else:
                if isinstance(default_value, str):
                    args[k] = v
                elif isinstance(default_value, bool):
                    args[k] = LITERAL_TO_BOOL[v]
                else:
                    args[k] = eval(v)
        else:
            args[k] = v
    return args
