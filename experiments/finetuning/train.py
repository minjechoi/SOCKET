import argparse
import json
import os
from os.path import join

os.environ["TOKENIZERS_PARALLELISM"]="false"

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers

from pl_models import SOCKETModule
from pl_data_loader import SOCKETDataModule
from SOCKET import TASK_DICT

def train(args):
    dict_args = vars(args)
    seed_everything(args.seed)
    # load logger
    print(os.getpid())
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.default_root_dir)

    dm = SOCKETDataModule(**dict_args)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="val_total_loss", mode="min",
        dirpath=args.default_root_dir,
        filename="checkpoint-{epoch}-{val_total_loss:.3f}",
        every_n_epochs=1, save_weights_only=True)

    early_stop_callback = EarlyStopping(
        monitor='val_total_loss', mode='min', min_delta=0.0, patience=3, verbose=False,
    )

    model = SOCKETModule(list_of_tasks=dm.list_of_tasks,
                         dataset_info=dm.dataset_info,
                         **dict_args)

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
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_mode", type=str, default=None)
    parser.add_argument("--multiple_seeds", type=int, default=None)
    parser = Trainer.add_argparse_args(parser)
    parser = SOCKETDataModule.add_model_specific_args(parser)
    parser = SOCKETModule.add_model_specific_args(parser)
    args = parser.parse_args()

    default_root_dir = args.default_root_dir
    if args.train_mode == 'all':
        for seed in range(1, 4):
            args.seed = seed
            args.default_root_dir = join(default_root_dir, f'seed-{seed}')
            list_of_tasks = []
            for category, V in TASK_DICT.items():
                list_of_tasks.extend(V)
            args.tasks = ','.join(list_of_tasks)
            train(args)

    elif args.train_mode == 'categorywise':
        for seed in range(1, 4):
            # categories = ['humor_sarcasm']
            categories = ['humor_sarcasm', 'offensive',
                          'sentiment_emotion', 'social_factors',
                          'trustworthy']
            args.seed = seed

            for cat in categories:
                list_of_tasks = TASK_DICT[cat]
                args.tasks = ','.join(list_of_tasks)
                args.default_root_dir = join(default_root_dir, f'seed-{seed}', 'all-' + cat)
                train(args)


    elif args.train_mode == 'single':
        for seed in range(1, 4):
            args.seed = seed
            list_of_tasks = []
            for k, V in TASK_DICT.items():
                list_of_tasks.extend(V)

            for task in list_of_tasks:
                args.tasks = task
                args.default_root_dir = join(default_root_dir, f'seed-{seed}', task)
                train(args)

    else:
        if args.do_train:
            train(args)

"""
CUDA_VISIBLE_DEVICES=0 python run.py --do_train --train_mode all \
  --default_root_dir /shared/3/projects/socket/experiments/camera-ready/ex1-deberta-all \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 --max_epochs 3 --precision 16 --seed 1 \
  --warmup_steps 0.06 --train_batch_size 16

CUDA_VISIBLE_DEVICES=2 python run.py --do_train --train_mode categorywise \
  --default_root_dir /shared/3/projects/socket/experiments/camera-ready/ex2-deberta-categorywise \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 --max_epochs 3 --precision 16 --seed=1 \
  --warmup_steps 0.06 --train_batch_size 16

CUDA_VISIBLE_DEVICES=6 python run.py --do_train --train_mode single \
  --default_root_dir /shared/3/projects/socket/experiments/ex3-deberta-single \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 --max_epochs 10 --precision 16 \
  --warmup_steps 0.06
  
CUDA_VISIBLE_DEVICES=6 python run.py --do_train --tasks=talkdown-pairs \
  --default_root_dir /shared/3/projects/socket/experiments//test \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu --devices 1 --max_epochs 10 --precision 16 \
  --warmup_steps 0.06
  
hasbiasedimplication,neutralizing-bias-pairs  
"""