import os
import argparse
from utils import (
    get_default_arg,
    add_argument_from_default_config,
    process_namespace_arguments,
)
from .hf_train import hyperparameter_optimization_with_trainer_api, objective

if __name__ == "__main__":

    all_config = get_default_arg()

    parser = argparse.ArgumentParser()

    add_argument_from_default_config(parser, all_config)

    parser.add_argument('--datadir', type=str, default=os.environ['SM_CHANNEL_DATADIR'])
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--sm_action", type=str)

    args, _ = parser.parse_known_args()
    input_args = process_namespace_arguments(args, all_config)

    print("*****Sagemaker Training*****")
    print(f"sm_action: {args.sm_action}\ndebug: {args.mode == 'debug'}")

    if args.sm_action == 'train':
        input_args = process_namespace_arguments(args, all_config, True)
        print(f"\n=====\nINPUT ARGUMENTS:\n{input_args}\n======\n")
        scores = objective(input_args)
        print(f"\n===== SCORES: {scores} =====\n")

    elif args.sm_action == 'hyper_optim':
        input_args = process_namespace_arguments(args, all_config, False)
        print(f"\n=====\nINPUT ARGUMENTS:\n{input_args}\n======\n")
        hyperparameter_optimization_with_trainer_api(input_args)

    else:
        print(f"\n=====\nINPUT ARGUMENTS:\n{input_args}\n======\n")
        print("No action taken")

    print("*****Complete*****")
