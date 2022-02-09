#!/usr/bin/env python3
import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import soundfile as sf
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('input_csv')
parser.add_argument('-o',
                    '--output',
                    type=str,
                    required=True,
                    help='Output data hdf5')
parser.add_argument('-c', type=int, default=1)
parser.add_argument('--use_fullname', '-u', action='store_true', default=False)
args = parser.parse_args()

df = pd.read_csv(args.input_csv, sep="\s+")
assert 'filename' in df.columns, "Header needs to contain 'filename'"


def read_wav_soundfile(fname):
    y, sr = sf.read(fname, dtype='int16')
    if y.ndim > 1:
        y = y[0]
    return y.astype('int16')


with h5py.File(args.output, 'w') as store:
    for fname in tqdm(df['filename'].unique(), unit='file'):
        feat = read_wav_soundfile(fname)
        if args.use_fullname:
            basename = fname
        else:
            basename = Path(fname).name
        try:
            store[basename] = feat
        except OSError:
            print(f"Warning, {fname} already exists!")
