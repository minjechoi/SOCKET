import argparse
import json
import os
from os.path import join
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers

from pl_models import SOCKETModule
from pl_data_loader import SOCKETDataModule
sys.path.append('../..')
from SOCKET import TASK_DICT

def eval(args):
    dict_args = vars(args)
    print(os.getpid())
    # the arguments used for creating the dataset are different from the ones used for loading the model
 
    # list_of_tasks = []
    # for category, V in TASK_DICT.items():
    #     list_of_tasks.extend(V)
    # all_tasks=','.join(list_of_tasks)
    tasks=args.tasks.split(',')
    # dict_args['tasks']=all_tasks
    dm = SOCKETDataModule(
        splits=['test'],
        **dict_args) # prediction for all tasks

    model = SOCKETModule(list_of_tasks=tasks,
                         dataset_info=dm.dataset_info,
                         **dict_args)

    trainer = Trainer.from_argparse_args(args)
    result = trainer.test(model, datamodule=dm, ckpt_path=args.ckpt_file_path)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(join(args.save_dir, args.save_file_name), 'w') as f:
        json.dump(result,f)
        # f.write(json.dumps(result))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments for prediction
    parser.add_argument("--ckpt_dir", type=str, help="File path for the checkpoint model directories")
    parser.add_argument("--eval_mode", type=str, choices=['all','categorywise','single'])
    parser.add_argument("--save_dir", type=str, help="File path for the saved data")
    parser.add_argument("--save_file_name", type=str, default='test-results.json', help='Use custom name to store test results as .json file')
    
    parser = Trainer.add_argparse_args(parser)
    parser = SOCKETDataModule.add_model_specific_args(parser)
    parser = SOCKETModule.add_model_specific_args(parser)
    args = parser.parse_args()
    
                
    assert args.save_dir is not None, "--save_dir should be set to a file path"
    if args.eval_mode=='all':
        list_of_tasks=[]
        for cat,tasks in TASK_DICT.items():
            list_of_tasks.extend(tasks)
        # get model directory
        model_file = [file for file in os.listdir(args.ckpt_dir) if file.endswith('.ckpt')][0]
        args.ckpt_file_path = join(args.ckpt_dir,model_file)
        args.tasks = ','.join(list_of_tasks) # used for the model
        eval(args)

    elif args.eval_mode=='categorywise':
        if args.tasks:
            categories = args.tasks.split(',')
            for cat in categories:
                assert cat in TASK_DICT, "Error! categories specified in --tasks does not fit into any of the 5 social categories: " + \
                    "[humor_sarcasm, offensive, sentiment_emotion, social_factors, trustworthy]"
        else:
            categories = ['humor_sarcasm', 'offensive',
                        'sentiment_emotion', 'social_factors',
                        'trustworthy']
        for cat in categories:
            list_of_tasks = TASK_DICT[cat]
            model_file = [file for file in os.listdir(join(args.ckpt_dir,cat)) if file.endswith('.ckpt')][0]
            args.ckpt_file_path = join(args.ckpt_dir,cat,model_file)
            args.tasks = ','.join(list_of_tasks) # used for the model
            eval(args)

    elif args.eval_mode=='single':
        if args.tasks:
            list_of_tasks = args.tasks.split(',')
        else:
            list_of_tasks = []
            for k, V in TASK_DICT.items():
                list_of_tasks.extend(V)
        print(list_of_tasks)
        for task in list_of_tasks:
            args.tasks = task
            model_file = [file for file in os.listdir(join(args.ckpt_dir,task)) if file.endswith('.ckpt')][0]
            args.ckpt_file_path = join(args.ckpt_dir,task,model_file)
            args.save_file_name = f'{task}.json'
            eval(args)
        

"""
# sushi
CUDA_VISIBLE_DEVICES=1 python eval.py --eval_mode single \
  --ckpt_dir /shared/3/projects/socket/experiments/camera-ready/ex3-deberta-single/seed-1 \
  --save_dir /shared/3/projects/socket/experiments/camera-ready/compare-socket-sockette/sockette-single \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 \
  --eval_batch_size 16 --use_sockette


CUDA_VISIBLE_DEVICES=2 python eval.py --eval_mode single \
  --ckpt_dir /shared/3/projects/socket/experiments/camera-ready/ex3-deberta-single/seed-1 \
  --save_dir /shared/3/projects/socket/experiments/camera-ready/compare-socket-sockette/socket-single \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 \
  --eval_batch_size 16
"""