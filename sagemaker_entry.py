import os
import argparse
from utils import read_json, add_argument_from_default_config
from segmentation import train
from hf_train import hyperparameter_optimization

if __name__ == "__main__":

#     config_files = ['segmentation_config.json', 'training_config.json']
#     configs = [read_json(c) for c in config_files]

    parser = argparse.ArgumentParser()

#     for config in configs:
#         add_argument_from_default_config(parser, config)

    parser.add_argument('--datadir', type=str, default=os.environ['SM_CHANNEL_DATADIR'])
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--sm_action", type=str, default='train')
    parser.add_argument("--mode", type=str, default='debug')

    args, _ = parser.parse_known_args()
    input_args = vars(args)

    print("*****Sagemaker Training*****\n")
    print(f"sm_action: {args.sm_action}\ndebug: {args.mode == 'debug'} ")
    print(f"ARGUMENTS:\n=====\n{args}\n======\n")

    if args.sm_action == 'train':
        train(**input_args)
    elif args.sm_action == 'hyper_optim':
        hyperparameter_optimization(input_args)
    else:
        print("No action taken")
    print("*****Complete*****")
