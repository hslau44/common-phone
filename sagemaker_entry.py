import os
import argparse
from utils import *
from segmentation import train


if __name__ == "__main__":
    
    config_files = ['segmentation_config.json','training_config.json',]
    configs = [read_json(c) for c in config_files]
    
    parser = argparse.ArgumentParser()
    
    for config in configs:
        add_argument_from_default_config(parser,config)
        
    parser.add_argument('--datadir', type=str, default=os.environ['SM_CHANNEL_DATADIR'])
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    
    args, _ = parser.parse_known_args()
    
    args_dic = extract_args_by_default_config(args,configs[0])
    args_dic['training_config'] = extract_args_by_default_config(args,configs[1])
    args_dic['output_data_dir'] = args.output_data_dir
    args_dic['datadir'] = args.datadir
    
    print("ARGS:\n",args_dic,"\n")
    train(**args_dic)