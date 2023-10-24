# Experiment: Fine-tuning

This directory contains the code for training, evaluating single-task and multi-task models and making predictions

## Training script
```python
python train.py 
  --train_mode all \
  --default_root_dir directory/to/save/model \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu \
  --devices 1 \
  --tasks tweet_emotion,crowdflower \
  --max_epochs 10
  --precision 16 \
  --warmup_steps 0.06 \
  --seed=1 \
  --n_seeds=1
```

- train_mode: Defines which type of training (single-task or multi-task).
    - single: Creates separate model for each of the specified tasks. A separate subdirectory is created for each task.
    - categorywise: Runs multi-task at a category level. A separate subdirectory is created for each category name.
    - all: Trains a single model that learns to solve all of the different tasks.

- tasks: If specified, only runs on that subset of tasks. To specify a list of tasks, return argument as all tasks separated by comma. Set to 'None' by default.
  - if 'train_mode' is set to 'categorywise', a subset of categories can be trained by specifying the list of categories (instead of tasks)
    - e.g. python train.py --train_mode categorywise --tasks social_factors,offensiveness

## Evaluation script
```python
python eval.py 
  --eval_mode all \
  --default_root_dir directory/to/save/model \
  --model_name_or_path microsoft/deberta-v3-base \
  --accelerator gpu \
  --devices 1 \
  --tasks tweet_emotion,crowdflower \
  ==use_sockette
```

- eval_mode: Defines which type of evaluation (single-task or multi-task).
    - single: Creates separate model for each of the specified tasks. A separate subdirectory is created for each task.
    - categorywise: Runs multi-task at a category level. A separate subdirectory is created for each category name.
    - all: Trains a single model that learns to solve all of the different tasks.

- tasks: If specified, only runs on that subset of tasks. To specify a list of tasks, return argument as all tasks separated by comma. Set to 'None' by default.
  - if 'train_mode' is set to 'categorywise', a subset of categories can be trained by specifying the list of categories (instead of tasks)
    - e.g. python train.py --train_mode categorywise --tasks social_fators,offensiveness

- use_sockette: If flag is used, evaluate on SocKETTe (a truncated version which only contains up to 1,000 samples per test set for faster evaluation) instead of the entire model.