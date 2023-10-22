# Zero-shot evaluations
This directory contains the scripts for evaluating SocKET using LLMs in a zero-shot manner.

```python
python predict.py --batch_size 32 --model_name_or_path decapoda-research/llama-13b-hf --use_sockette --result_path predictions_sockette --use_cuda --tasks CLS

## Arguments
 --tasks : Can specify list of tasks to use. Available options: (1) ALL (set as default), (2) type of tasks (from [CLS, SPAN, PAIR, REG]), (3) list of specific tasks, should be divided by ',' (e.g., toxic-span,tweet_emoji,emobank#dominance)
 --model_name_or_path : Name of Hugging Face repository of model. Make sure to specify in the predict.py script if a 'text-generation' or 'text2text-generation' pipeline should be applied.
 --result_path : Directory of where to store prediction results
 --batch_size : Batch size to run evaluation
 --use_sockette : Whether to evaluate on SocKETTe instead of the entire test set, which will evaluate on max 1000 samples per task.
 --use_cuda : Whether to run code on CUDA (requires GPU access)
```