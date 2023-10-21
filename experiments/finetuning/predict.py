import argparse
import json
import os
from os.path import join
import sys

os.environ["TOKENIZERS_PARALLELISM"]="false"

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers

from pl_models import SOCKETModule
from pl_data_loader import SOCKETDataModule
sys.path.append('../..')
from SOCKET import TASK_DICT

def predict(args):
    dict_args = vars(args)
    print(os.getpid())
    # the arguments used for creating the dataset are different from the ones used for loading the model
 
    list_of_tasks = []
    for category, V in TASK_DICT.items():
        list_of_tasks.extend(V)
    all_tasks=','.join(list_of_tasks)
    model_tasks=dict_args['tasks']
    dict_args['tasks']=all_tasks
    dm = SOCKETDataModule(
      splits=['test'],
      **dict_args) # prediction for all tasks

    model = SOCKETModule(list_of_tasks=model_tasks.split(','),
                         dataset_info=dm.dataset_info,
                         **dict_args)

    trainer = Trainer.from_argparse_args(args)
    results = trainer.predict(model, datamodule=dm, ckpt_path=args.ckpt_file_path)

    # get all tasks
    all_scores={}
    all_tasks=[]
    for res,data_tasks in results:
        all_tasks.extend(data_tasks)
        for model_task,V in res.items():
            if model_task not in all_scores:
                all_scores[model_task]=[]
            all_scores[model_task].extend(V)
    all_tasks = [list_of_tasks[i] for i in all_tasks]

    if not os.path.exists(args.predict_save_dir):
        os.makedirs(args.predict_save_dir)
    for model_task,V in all_scores.items():
        with open(join(args.predict_save_dir,f'{model_task}.tsv'),'w') as f:
            for data_task,arr in zip(all_tasks,V):
                f.write('\t'.join([data_task,','.join([str(round(x,3)) for x in arr])])+'\n')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments for prediction
    parser.add_argument("--ckpt_dir", type=str, help="File path for the checkpoint model directories")
    parser.add_argument("--predict_mode", type=str, choices=['all','categorywise','single'])
    parser.add_argument("--predict_save_dir", type=str, help="File path for the saved data")
    
    parser = Trainer.add_argparse_args(parser)
    parser = SOCKETDataModule.add_model_specific_args(parser)
    parser = SOCKETModule.add_model_specific_args(parser)
    args = parser.parse_args()
    
                
    assert args.predict_save_dir is not None, "--predict_save_dir should be set to a file path"
    if args.predict_mode=='all':
        list_of_tasks=[]
        for cat,tasks in TASK_DICT.items():
            list_of_tasks.extend(tasks)
        # get model directory
        model_file = [file for file in os.listdir(args.ckpt_dir) if file.endswith('.ckpt')][0]
        args.ckpt_file_path = join(args.ckpt_dir,model_file)
        args.tasks = ','.join(list_of_tasks) # used for the model
        predict(args)

    elif args.predict_mode=='categorywise':
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
            predict(args)

    elif args.predict_mode=='single':
        if args.tasks:
            list_of_tasks = args.tasks.split(',')
        else:
            list_of_tasks = []
            for k, V in TASK_DICT.items():
                list_of_tasks.extend(V)
        for task in list_of_tasks:
            args.tasks = task
            model_file = [file for file in os.listdir(join(args.ckpt_dir,task)) if file.endswith('.ckpt')][0]
            args.ckpt_file_path = join(args.ckpt_dir,task,model_file)
            predict(args)
        
        
"""
# sushi
CUDA_VISIBLE_DEVICES=1 python predict.py --predict_mode single \
  --ckpt_dir /shared/3/projects/socket/experiments/camera-ready/ex3-deberta-single/seed-3 \
  --tasks=emotion-span \
  --predict_save_dir /shared/3/projects/socket/experiments/camera-ready/ex3-deberta-single/seed-3 \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 \
  --eval_batch_size 16

CUDA_VISIBLE_DEVICES=2 python run.py --do_train --train_mode all \
  --default_root_dir /shared/3/projects/socket/experiments/camera-ready/ex1-deberta-all \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 --max_epochs 5 --precision 16 --seed 2 --n_seeds=1 \
  --warmup_steps 0.06 --train_batch_size 16

CUDA_VISIBLE_DEVICES=5 python run.py --do_train --train_mode all \
  --default_root_dir /shared/3/projects/socket/experiments/camera-ready/ex1-deberta-all \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 --max_epochs 5 --precision 16 --seed 3 --n_seeds=1 \
  --warmup_steps 0.06 --train_batch_size 16

CUDA_VISIBLE_DEVICES=6 python run.py --do_train --train_mode categorywise --task=offensive,sentiment_emotion,trustworthy \
  --default_root_dir /shared/3/projects/socket/experiments/camera-ready/ex2-deberta-categorywise \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 --max_epochs 3 --precision 16 --seed=1 --n_seeds=2 \
  --warmup_steps 0.06 --train_batch_size 16

  



# taco
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --train_mode single --task=emotion-span,toxic-span,propaganda-span \
  --default_root_dir /shared/3/projects/socket/experiments/camera-ready/ex3-deberta-single \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 --max_epochs 10 --precision 16 \
  --warmup_steps 0.06 --seed=1 --n_seeds=1
  

CUDA_VISIBLE_DEVICES=0 python run.py --do_train --train_mode categorywise --task=offensive,sentiment_emotion,trustworthy,social_factors \
  --default_root_dir /shared/3/projects/socket/experiments/camera-ready/ex2-deberta-categorywise \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 --max_epochs 3 --precision 16 --seed=3 --n_seeds=1 \
  --warmup_steps 0.06 --train_batch_size 16
  

CUDA_VISIBLE_DEVICES=1 python run.py --do_train --train_mode single --tasks=emotion-span,toxic-span,propaganda-span \
  --default_root_dir /shared/3/projects/socket/experiments/camera-ready/ex3-deberta-single \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 --max_epochs 10 --precision 16 \
  --warmup_steps 0.06 --seed=2 --n_seeds=2

CUDA_VISIBLE_DEVICES=2 python run.py --do_train --train_mode single \
  --default_root_dir /shared/3/projects/socket/experiments/camera-ready/ex3-deberta-single \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 --max_epochs 10 --precision 16 \
  --warmup_steps 0.06 --seed=3 --n_seeds=1




"""