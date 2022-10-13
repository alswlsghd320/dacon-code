import os 
import math
import random
import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

from config import getConfig

import warnings
warnings.filterwarnings('ignore')

def train(args):
    print('<---- Training Params ---->')
    pprint.pprint(args)

    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    model = CrossEncoder(args.pretrained, num_labels=1)
    model.tokenizer.truncation_side = 'left'
    print(model)

    train_df = pd.read_csv(f"./data/train_{args.fold}fold.csv")
    val_df = pd.read_csv(f"./data/val_{args.fold}fold.csv")

    train_samples = []

    for _, row in tqdm(train_df.iterrows()):
        train_samples.append(InputExample(texts = [row['code1'], row['code2']], label = row['similar']))
        train_samples.append(InputExample(texts = [row['code2'], row['code1']], label = row['similar']))

    dev_samples = []

    for _, row in tqdm(val_df.iterrows()):
        dev_samples.append(InputExample(texts = [row['code1'], row['code2']], label = row['similar']))

    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)


    # We add an evaluator, which evaluates the performance during training
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='')
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.1)

    # Train the model
    model.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=args.epochs,
            evaluation_steps=10000,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
            use_amp=True,
            save_best_model=True,
            #checkpoint_path=os.path.join(args.save_path, args.tag),
            output_path=os.path.join('results', args.tag),
            #checkpoint_save_total_limit=5,
            )

if __name__ == '__main__':
    args = getConfig()
    train(args)