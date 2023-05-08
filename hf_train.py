import os
import copy
import datetime
import joblib
import optuna
import transformers, torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback, 
    ProgressCallback,
    TensorBoardCallback,
)
from segmentation import (
    PhonemeDetailsDataset, 
    PhonemeSegmentor, 
    TrainingDataProcessor, 
    get_metadata, 
    CustomTrainer,
    compute_avg_sample_acc,
)
from models import CustomWav2Vec2Segmentation
from utils import read_json, get_default_arg


# compute config
transformers.utils.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = torch.cuda.is_available()
# args to be feed to TrainingArguments
ACCEPT_TRAINARGS = [
    'seed',
    'num_train_epochs',
    'per_device_train_batch_size',
    'per_device_eval_batch_size',
    'evaluation_strategy',
    'save_strategy',
    'logging_strategy',
    'save_steps',
    'eval_steps',
    'logging_steps',
    'learning_rate',
    'adam_beta1',
    'adam_beta2',
    'adam_epsilon',
    'weight_decay',
    'lr_scheduler_type',
    'warmup_steps',
    'optim',
    'gradient_checkpointing',
    'gradient_accumulation_steps',
    'fp16',
    'tf32',
    'group_by_length',
    'remove_unused_columns',
    'save_total_limit',
    'push_to_hub'
    'metric_for_best_model',
    'greater_is_better',
    'load_best_model_at_end',
]


def hp_space(trial):

    args = {
        'per_device_train_batch_size': trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8]),
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        'warmup_steps': trial.suggest_float("warmup_steps", 1, 1000, log=True),
        'optim': trial.suggest_categorical("optim", ["adafactor", "adamw_torch"]),
    }

    args.update({
        'resolution': trial.suggest_categorical("resolution", [0.005, 0.01, 0.02]),
        't_end': 8,
        'pad_to_sec': 1,
        'num_encoders': None, # trial.suggest_int("num_encoders", 5, 12), # 
        'num_convprojs': trial.suggest_int("num_convprojs", 1, 5),
        'conv_hid_actv': trial.suggest_categorical("conv_hid_actv", ["gelu", "relu","none"]),
        'conv_last_actv': None,
        'freeze_encoder': True,
    })
    
    
    
    # TODO add args
    args['num_labels'] = 121
    args['metric_for_best_model'] = 'eval_loss'
    args['greater_is_better'] = False
    args['early_stopping_patience'] = 3
    args['early_stopping_threshold'] = 0.0
    args['load_best_model_at_end'] = True
    
#     # local setting
#     args['datadir'] = 'data'
#     args['output_data_dir'] = 'outputs/exp_04'
#     args['tf32'] = False
#     args['per_device_train_batch_size'] = 2
#     args['per_device_eval_batch_size'] = 2
    
    return args


def set_training_arguments(args):
    training_args = {}
    for k,v in args.items():
        if k in ACCEPT_TRAINARGS:
            training_args[k] = v
    
    if 'model_dir' in args.keys():
        training_args.update(
            output_dir=args['model_dir'],
            logging_dir=args['output_data_dir'],
        )
    else:
        training_args.update(
            output_dir=args['output_data_dir'],
            logging_dir=None,
        )
    return training_args


def get_datetime():
    k = datetime.datetime.now()
    t = str(k.date()) + str('_') + str(k.hour) + str('-') + str(k.minute)
    return t


def get_callbacks(args):
    patient = args.get('early_stopping_patience',1)
    threshold = args.get('early_stopping_threshold',0.0)
    callbacks = [
        EarlyStoppingCallback(patient,threshold),
        ProgressCallback(),
        TensorBoardCallback(),
    ]
    return callbacks


def objective(args):
    
    trainset = PhonemeDetailsDataset_(_set='train',**args)
    validaset = PhonemeDetailsDataset_(_set='dev',**args)
    
    data_collator = TrainingDataProcessor_(**args)
    
    model = CustomWav2Vec2Segmentation(**args)
    
    training_args = TrainingArguments(**set_training_arguments(args))
    
    callbacks = get_callbacks(args)
    
    trainer = CustomTrainer(
        model=model,
        data_collator=data_collator, 
        args=training_args,
        compute_metrics=compute_avg_sample_acc, #
        train_dataset=trainset,
        eval_dataset=validaset,
        callbacks=callbacks,
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    score = eval_results['eval_avg_sample_acc']
    return score


class PhonemeDetailsDataset_(PhonemeDetailsDataset):
    
    def __init__(self,_set,mode,**kwargs):
        data_dir = kwargs['datadir'] # <------- find correct arg name
        path = os.path.join(data_dir,'metadata.csv')
        if _set == 'test':
            _locale = kwargs['test_locales']
        else:
            _locale = kwargs['train_locales']
        metadata = get_metadata(path=path,_set=_set,_locale=_locale)
        if mode == 'debug':
            metadata = metadata.copy().iloc[:64,:]
            metadata.reset_index(drop=True, inplace=True)
        super().__init__(metadata,data_dir)


class TrainingDataProcessor_(TrainingDataProcessor):
    
    def __init__(self,model_checkpoint,sampling_rate,resolution,t_end,pad_to_sec,**kwargs):
        # model and data-processor config 
        hf_config = AutoConfig.from_pretrained(model_checkpoint)
        tokenizer_type = hf_config.model_type if hf_config.tokenizer_class is None else None
        hf_config = hf_config if hf_config.tokenizer_class is not None else None

        pad_token="(...)"
        unk_token="UNK"
        tokenizer = AutoTokenizer.from_pretrained(
          "./",
          config=hf_config,
          tokenizer_type=tokenizer_type,
          unk_token=unk_token,
          pad_token=pad_token,
        )

        segmentor = PhonemeSegmentor(tokenizer=tokenizer,resolution=resolution,pad_token=pad_token)
        
        super().__init__(
            sampling_rate=sampling_rate,
            resolution=resolution, 
            t_end=t_end, 
            pad_to_sec=pad_to_sec, 
            tokenizer=segmentor 
        )


class Objective_class(object):
    
    def __init__(self, local_args):
        self.default_args = get_default_arg()
        self.local_args = local_args
    
    def __call__(self,trial):
        args = self.set_hp_space(trial)
        score = objective(args)
        return score 
    
    def set_hp_space(self,trial):
        args = copy.deepcopy(self.default_args)
        args.update(hp_space(trial))
        args.update(self.local_args)
        args = self.exp_code_ops(args)
        return args
    
    def exp_code_ops(self,args):
        exp_code = get_datetime()
        args['output_data_dir'] = os.path.join(args['output_data_dir'],exp_code)
        if 'model_dir' in args.keys():
            args['model_dir'] = os.path.join(args['model_dir'],exp_code)
        return args


class HyperOptHelper:
    
    def __init__(self,local_args):
        self.default_args = get_default_arg()
        self.local_args = local_args
        
    def hp_space(self,trial):
        
        args = copy.deepcopy(self.default_args)
        args.update({
            'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            'warmup_steps': trial.suggest_float("warmup_steps", 1, 1000, log=True),
            'optim': trial.suggest_categorical("optim", ["adafactor", "adamw_torch"]),
        })
        args.update(self.local_args)
        
        return args
    
    def model_init(self,trial):
        
        args = copy.deepcopy(self.default_args)
        args.update({
            'resolution': trial.suggest_categorical("resolution", [0.005, 0.01, 0.02]),
            'num_encoders': None, # trial.suggest_int("num_encoders", 5, 12),  
            'num_convprojs': trial.suggest_int("num_convprojs", 1, 5),
            'conv_hid_actv': trial.suggest_categorical("conv_hid_actv", ["gelu", "relu","none"]),
            'freeze_encoder': True
        })
        args.update(self.local_args)
        
        model = CustomWav2Vec2Segmentation(**args)
        return model
    
    def get_updated_non_tunable_arguments(self):
        args = copy.deepcopy(self.default_args)
        args.update(self.local_args)
        return args

    
def hyperparameter_optimization(args):
    """
    Run hpyerparameter optimization
    Arguments in 'args' will overwrite the arguments generated 
    from function 'hp_space', which means they will no longer
    be tunable
    """
    optuna_objective = Objective_class(args)
    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_objective, n_trials=args['n_trials'])
    study_fp = os.path.join(args['output_data_dir'],'study.pkl')
    joblib.dump(study, study_fp)
    return


def hyperparameter_optimization_with_trainer_api(args):
    

    
    helper = HyperOptHelper(args)
    ntune_args = helper.get_updated_non_tunable_arguments()
    trainset = PhonemeDetailsDataset_(_set='train',**ntune_args)
    validaset = PhonemeDetailsDataset_(_set='dev',**ntune_args)
    data_collator = TrainingDataProcessor_(**ntune_args)
    
    training_args = TrainingArguments(**set_training_arguments(args))
    callbacks = callbacks = get_callbacks(args)
    
    trainer = CustomTrainer(
        model_init=helper.model_init,
        data_collator=data_collator, 
        args=training_args,
        compute_metrics=compute_avg_sample_acc, # import
        train_dataset=trainset,
        eval_dataset=validaset,
        callbacks=callbacks,
    )
    
    # as CustomTrainer is used, (aggregated) metric, i.e. avg_sample_acc is used, 
    # thus direction="maximize"
    
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=helper.hp_space,
        n_trials=args['n_trials'],
    )
    print(best_trial)
    return


if __name__ == "__main__":
    # local setting
    args['mode'] = 'train'
    args['datadir'] = 'data'
    args['output_data_dir'] = 'outputs/exp_05'
    args['tf32'] = False
    args['per_device_train_batch_size'] = 2
    args['per_device_eval_batch_size'] = 2
    args['num_train_epochs'] = 40
    args['n_trials'] = 16
    
    hyperparameter_optimization(args)
    
    