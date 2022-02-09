import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict,List
import torchaudio.transforms as audio_transforms
from einops import rearrange
import torch_audiomentations as wavtransforms
import torch
import yaml
# Some defaults for non-specified arguments in yaml
DEFAULT_TRAIN_ARGS = {
    'outputpath': 'experiments',
    'train_data': 'data/labels/balanced.csv',
    'cv_data': 'data/labels/balanced.csv',
    'test_data': 'data/labels/eval.csv',
    'loss': 'BCELoss',  # default is BCEloss.
    'loss_args': {},
    'student': 'MobileNetV2_DM',
    'student_args': {},
    'batch_size': 32,
    'warmup_iters': 1000,
    'max_grad_norm': 2.0,
    'mixup': None,
    'epoch_length': None,
    'num_workers': 3,  # Number of dataset loaders
    'spectransforms': {},  #Default no augmentation
    'wavtransforms': {},
    'early_stop': 5,
    'epochs': 120,
    'n_saved': 4,
    'optimizer': 'Adam',
    'optimizer_args': {
        'lr': 0.001,
    },
}
DEFAULT_CHUNK_ARGS = {
    'outputpath': 'experiments',
    'train_data': 'data/labels/balanced.csv',
    'cv_data': 'data/labels/balanced.csv',
    'test_data': 'data/labels/eval.csv',
    'loss': 'BCELoss',  # default is BCEloss.
    'loss_args': {},
    'batch_size': 32,
    'warmup_iters': 1000,
    'max_grad_norm': 2.0,
    'mixup': None,
    'epoch_length': None,
    'num_workers': 3,  # Number of dataset loaders
    'spectransforms': {},  #Default no augmentation
    'wavtransforms': {},
    'early_stop': 5,
    'epochs': 120,
    'n_saved': 1,
    'optimizer': 'Adam',
    'optimizer_args': {
        'lr': 0.001,
        'amsgrad':True,
    },
}

def parse_config_or_kwargs(config_file, default_args = DEFAULT_TRAIN_ARGS, **override_kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from config file are all possible params
    arguments = dict(yaml_config, **override_kwargs)
    # In case some arguments were not passed, replace with default ones
    for key, value in default_args.items():
        arguments.setdefault(key, value)
    return arguments

def _pd_cast_to_int(inputarr):
    return np.array(inputarr, dtype=int)


def read_tsv_data_chunked(data_path: str,
                          chunk_length: int = 1, # The length for each chunk
                          chunk_hop: int = None, # The hop size for each chunk
                          nrows: int = None,
                          basename: bool = True) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep='\t', nrows=nrows, dtype={
        'labels': str
    }).dropna()  #drops some indices during evaluation which have no labels
    if 'labels' in df.columns and not pd.api.types.is_numeric_dtype(
            df['labels']):
        df['labels'] = df['labels'].str.split(';').apply(
            lambda x: np.array(x, dtype=int))
    if chunk_hop == None:
        chunk_hop = chunk_length

    df['from'] = df['duration'].apply(lambda x: np.arange(0, x, chunk_hop))
    df = df.explode('from')
    # Maximum between max duration and chunk lengths as duration
    df['to'] = np.minimum(df['from'] + chunk_length, df['duration'])

    # Get basename instead of abspath
    # This is generally not necessary except for cases where data samples have the same name
    # but are in different directories
    if basename:  
        df['filename'] = df['filename'].str.split('/').str[-1]
    return df.reset_index(drop=True)  # In case index has been modified


def read_tsv_data(path: str, nrows: int = None, basename: bool = True):
    df = pd.read_csv(path, sep='\t', nrows=nrows).dropna(
    )  #drops some indices during evaluation which have no labels
    if 'labels' in df.columns:
        df['labels'] = df['labels'].str.split(';').apply(_pd_cast_to_int)
    #Stronk data
    if ('labels' in df.columns) and ('onset' in df.columns) and (
            'offset' in df.columns) and ('hdf5path' in df.columns):
        # Aggregate to filename -> [[on1, on2],[off1, off2], [l1,l2]]
        df = df.groupby('filename').agg({
            'onset': list,
            'offset': list,
            'event_label': list,
            'hdf5path': lambda x: x.iloc[0],
        }).reset_index()
    if basename:  # Get basename instead of abspath
        df['filename'] = df['filename'].str.split('/').str[-1]
    return df.reset_index(drop=True)  # In case index has been modified


def mixup(x: torch.Tensor, lamb: torch.Tensor):
    """
    Mixes even with odds elements in x using weights in lamb.
    
    x: (batch, ndim)
    lamb: (batch/2)

    """

    if x.shape[0] % 2 != 0:
        return x
    x1, x2 = rearrange(x, '(b h) ... -> h ... b', h=2)
    mixed = x1 * lamb + x2 * (1. - lamb)
    return rearrange(mixed, '... b -> b ...')


class DictWrapper(object):
    def __init__(self, adict):
        self.dict = adict

    def state_dict(self):
        return self.dict

    def load_state_dict(self, state):
        self.dict = state

class PolyDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 decay_steps,
                 final_lrs,
                 power: float = 2,
                 last_epoch=-1,
                 verbose=False):
        self.decay_steps = decay_steps
        self.final_lrs = final_lrs
        self.power = power
        if not isinstance(self.final_lrs, list) and not isinstance(
                self.final_lrs, tuple):
            self.final_lrs = [self.final_lrs] * len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch, verbose)

    def _get_closed_form_lr(self):
        step = min(self.last_epoch + 1, self.decay_steps)
        current_lrs = []
        for base_lr, final_lr in zip(self.base_lrs, self.final_lrs):
            current_lr = ((base_lr - final_lr) *
                    (1 - step / self.decay_steps) ** (self.power)
                    ) + final_lr
            current_lrs.append(current_lr)
        return current_lrs

    def get_lr(self):
        return self._get_closed_form_lr()


def parse_wavtransforms(transforms_dict: Dict):
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    transforms = []
    for trans_name, v in transforms_dict.items():
        transforms.append(getattr(wavtransforms, trans_name)(**v))
    return torch.nn.Sequential(*transforms)

def parse_spectransforms(transforms_list: List):
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    return torch.nn.Sequential(*[
        getattr(audio_transforms, trans_name)(**v)
        for item in transforms_list
        for trans_name, v in item.items()
    ])
