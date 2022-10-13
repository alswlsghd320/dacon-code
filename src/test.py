import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder

import warnings
warnings.filterwarnings('ignore')

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/code', type=str)
    parser.add_argument('--pretrained', type=str, default='microsoft/graphcodebert-base')  #codebert-base
    parser.add_argument('--csv_name', default='sub2.csv', type=str)
    parser.add_argument('--threshold', default=0.5, type=float)
    args = parser.parse_args()

    return args

def preprocess_script_csv(c):
    lines = c.split('\n')
    preproc_lines = []
    for line in lines:
        if line.lstrip().startswith('#'):
            continue
        line = line.rstrip()
        if '#' in line and '"#"' not in line and "'#'" not in line:
            line = line[:line.index('#')]
        line = line.replace('\n','')
        line = line.replace('    ','\t')
        if line == '':
            continue
        preproc_lines.append(line)
    return '\n'.join(preproc_lines)

def test(args):
    test = pd.read_csv('data/test.csv')
    test['code1'] = test['code1'].apply(preprocess_script_csv)
    test['code2'] = test['code2'].apply(preprocess_script_csv)

    model = CrossEncoder(args.pretrained, num_labels=1)
    model.tokenizer.truncation_side = 'left'

    test_samples = []
    for _, row in tqdm(test.iterrows()):
        test_samples.append([row['code1'], row['code2']])

    with torch.no_grad():
        preds = [model.predict(sample) for sample in tqdm(test_samples)]
        
    sub = pd.read_csv('data/sample_submission.csv')
    sub['similar'] = preds
    sub['similar'] = sub['similar'].apply(lambda x : 0 if x < args.threshold else 1)

    save_path = os.path.join('./submission', args.csv_name)
    sub.to_csv(save_path,index=False)
    
    return preds

if __name__ == '__main__':
    args = getConfig()
    preds = test(args)
    npy_path = os.path.join('npys', args.csv_name[:-3]+'npy')
    np.save(npy_path, preds)