import argparse
import json
import os
from os.path import join
import sys

os.environ["TOKENIZERS_PARALLELISM"]="false"

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers

from pl_models import SOCKETModule
from pl_data_loader import SOCKETDataModule
sys.path.append('../..')
from SOCKET import TASK_DICT

def train(args):
    dict_args = vars(args)
    seed_everything(args.seed)
    # load logger
    print(os.getpid())
    if os.path.exists(args.default_root_dir):
      files = [file for file in os.listdir(args.default_root_dir) if file.endswith('.ckpt')]
      if len(files):
        print("Training ongoing! Terminating...")
        return
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.default_root_dir)

    dm = SOCKETDataModule(
      splits=['train','test','validation'],
      **dict_args)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, 
        monitor="val_total_score", mode="max",
        dirpath=args.default_root_dir,
        filename="checkpoint-{epoch}-{val_total_score:.3f}-{val_total_loss:.3f}",
        every_n_epochs=1, save_weights_only=True)

    early_stop_callback = EarlyStopping(
        monitor="val_total_score", mode="max",
        min_delta=0.0, patience=5, verbose=False,
    )

    model = SOCKETModule(
      list_of_tasks=dm.list_of_tasks,
      dataset_info=dm.dataset_info,
      **dict_args
    )

    trainer = Trainer.from_argparse_args(args, logger=tb_logger,
                                         callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, dm)

    result = trainer.test(model,
                          datamodule=dm, ckpt_path='best')[0]
    with open(join(args.default_root_dir, 'test-results.json'), 'w') as f:
        f.write(json.dumps(result))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments for train
    parser.add_argument("--train_mode", type=str, default=None, choices=['all','categorywise','single'])
    parser.add_argument("--exclude_tasks", type=str, default=None)
    # seeds
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    
    parser = Trainer.add_argparse_args(parser)
    parser = SOCKETDataModule.add_model_specific_args(parser)
    parser = SOCKETModule.add_model_specific_args(parser)
    args = parser.parse_args()
    
    default_root_dir = args.default_root_dir # directory to save models and scores
    if args.train_mode == 'all':
        list_of_tasks = []
        for category, V in TASK_DICT.items():
            list_of_tasks.extend(V)
        args.tasks = ','.join(list_of_tasks)
        for seed in range(args.seed, args.seed + args.n_seeds):
            args.seed = seed
            args.default_root_dir = join(default_root_dir, f'seed-{seed}')
            train(args)

    elif args.train_mode == 'categorywise':
      if args.tasks:
          categories = args.tasks.split(',')
          for cat in categories:
              assert cat in TASK_DICT, "Error! categories specified in --tasks does not fit into any of the 5 social categories: " + \
                  "[humor_sarcasm, offensive, sentiment_emotion, social_factors, trustworthy]"
      else:
          categories = ['humor_sarcasm', 'offensive',
                      'sentiment_emotion', 'social_factors',
                      'trustworthy']

      for seed in range(args.seed, args.seed + args.n_seeds):
          args.seed = seed            
          for cat in categories:
              list_of_tasks = TASK_DICT[cat]
              args.tasks = ','.join(list_of_tasks)
              args.default_root_dir = join(default_root_dir, f'seed-{seed}', 'all-' + cat)
              train(args)

    elif args.train_mode == 'single':
      list_of_tasks = []
      if args.tasks:
          list_of_tasks = args.tasks.split(',')
      else:
          for k, V in TASK_DICT.items():
              list_of_tasks.extend(V)
      if args.exclude_tasks:
        exclude_tasks = args.exclude_tasks.split(',')
      else:
        exclude_tasks = []
        
      for seed in range(args.seed, args.seed + args.n_seeds):
          args.seed = seed
          for task in list_of_tasks:
            if task in exclude_tasks:
              continue
            else:
              args.tasks = task
              args.default_root_dir = join(default_root_dir, f'seed-{seed}', task)
              train(args)