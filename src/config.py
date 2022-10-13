import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/code', type=str)
    parser.add_argument('--save_path', default='./checkpoints', type=str)
    parser.add_argument('--logging_path', default='./logs', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--use_tensorboard', default=True, type=bool)
    parser.add_argument('--tag', default='', type=str)

    parser.add_argument('--initial_lr', default=2e-05, type=float)
    parser.add_argument('--seed', type=int, default=42)  
    parser.add_argument('--fold', default=0, type=int)   
    parser.add_argument('--kfold', default=5, type=int)
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--pretrained', type=str, default='microsoft/graphcodebert-base')  #codebert-base
 
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--patience', default=3, type=int)
    
    args = parser.parse_args()

    return args 