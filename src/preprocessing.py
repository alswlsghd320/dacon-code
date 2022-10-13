import os
import time
import argparse
from functools import partial
from multiprocessing import Pool
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/code', type=str)
    parser.add_argument('--save_path', default='data', type=str)
    parser.add_argument('--pretrained', type=str, default='microsoft/graphcodebert-base')
    parser.add_argument('--topk', default=5, type=int)   
    parser.add_argument('--ratio', default=0.2, type=float) 
    parser.add_argument('--kfold', default=5, type=int)
    parser.add_argument('--num_process', default=16, type=int)           
    args = parser.parse_args()

    return args    

def preprocess_script(script):
    '''
    간단한 전처리 함수
    주석 -> 삭제
    '    '-> tab 변환
    다중 개행 -> 한 번으로 변환
    '''
    with open(script,'r',encoding='utf-8') as file:
        lines = file.readlines()
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
        preprocessed_script = '\n'.join(preproc_lines)
    return preprocessed_script

def get_positive_pairs(df, problem, ratio=0.2):
    solution_codes = df[df['problem_num'] == problem]['code']

    # Get positive pairs
    positive_pairs = list(combinations(solution_codes.to_list(),2))
    length = int(len(positive_pairs) * ratio)
    np.random.shuffle(positive_pairs)

    return positive_pairs[:length]

def get_negative_pair(dfnp, bm25, topk, row):
    code = row[0]
    num = int(row[1][-3:])
    negative_code_scores = bm25.get_scores(code).argsort()[::-1]
    # 같은 문제 제거한 후 topk 추출
    negative_code_ranking = [i for i in negative_code_scores if num != int(dfnp[i][1][-3:])][:topk]
    # 이전에 중복된 pair 삭제
    negative_pairs = [(code, dfnp[i][0]) for i in negative_code_ranking if num < int(dfnp[i][1][-3:])]

    # If list does not empty
    if negative_pairs:
        return negative_pairs

def make_df(args, df, tokenizer, problem_nums, save_path):
    tokenized_corpus = [tokenizer.tokenize(code, truncation=True) for code in df['code'].to_numpy()]
    problems = sorted(list(set(problem_nums)))
    bm25 = BM25Okapi(tokenized_corpus)

    total_positive_pairs = []
    total_negative_pairs = []

    for problem in tqdm(problems):
        total_positive_pairs.extend(get_positive_pairs(df, problem, ratio=args.ratio))

    #total_negative_pairs = get_negative_pairs(df, bm25, topk=args.topk)

    pool = Pool(processes=args.num_process)
    dfnp = df.to_numpy()

    func = partial(get_negative_pair, dfnp, bm25, args.topk)
    results = pool.map(func, dfnp)
    pool.close()
    pool.join()

    for negative_pairs in results:
        if negative_pairs is not None:
            total_negative_pairs.extend(negative_pairs)

    pos_code1 = list(map(lambda x:x[0],total_positive_pairs))
    pos_code2 = list(map(lambda x:x[1],total_positive_pairs))

    neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
    neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

    pos_label = [1]*len(pos_code1)
    neg_label = [0]*len(neg_code1)

    pos_code1.extend(neg_code1)
    total_code1 = pos_code1
    pos_code2.extend(neg_code2)
    total_code2 = pos_code2
    pos_label.extend(neg_label)
    total_label = pos_label
    pair_data = pd.DataFrame(data={
        'code1':total_code1,
        'code2':total_code2,
        'similar':total_label
    })
    pair_data = pair_data.sample(frac=1).reset_index(drop=True)

    pair_data.to_csv(save_path,index=False)

if __name__ == '__main__':
    args = getConfig()
    problem_folders = os.listdir(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    tokenizer.truncation_side = 'left'

    preproc_scripts = []
    problem_nums = []

    print(f'--------------init--------------')

    for problem_folder in tqdm(problem_folders):
        scripts = os.listdir(os.path.join(args.data_path,problem_folder))
        problem_num = scripts[0].split('_')[0]
        for script in scripts:
            script_file = os.path.join(args.data_path,problem_folder,script)
            preprocessed_script = preprocess_script(script_file)

            preproc_scripts.append(preprocessed_script)
        problem_nums.extend([problem_num]*len(scripts))

    df = pd.DataFrame(data = {'code':preproc_scripts, 'problem_num':problem_nums})

    kf = StratifiedKFold(n_splits=args.kfold)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, y=df['problem_num'])):
        df.loc[val_idx, 'fold'] = fold
    
    for fold in range(args.kfold):
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]

        train_save_path = os.path.join(args.save_path, f'train_{fold}.csv')
        val_save_path = os.path.join(args.save_path, f'val_{fold}.csv')
        start = time.time()
        make_df(args, train_df, tokenizer, problem_nums, save_path=train_save_path)
        make_df(args, val_df, tokenizer, problem_nums, save_path=val_save_path)
        end = time.time()
        print(f'--------------{fold} fold finished : {(end-start)/60:.3f} minutes--------------')
