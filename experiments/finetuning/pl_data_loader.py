import re
import sys

import numpy as np
from datasets import load_dataset
from datasets.features.features import Value,ClassLabel,Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pytorch_lightning import LightningDataModule

sys.path.append('../../')
from SOCKET import TASK_DICT

task2cat={}
for category,tasks in TASK_DICT.items():
    for task in tasks:
        task2cat[task]=category

class DefaultDataset(Dataset):
    def __init__(self,texts,labels,tasks,task_types):
        self.texts=texts
        self.labels=labels
        self.tasks=tasks
        self.task_types=task_types

        print(len(texts),len(labels),len(tasks),len(task_types))

        assert len(texts)==len(labels)
        assert len(texts)==len(tasks)
        assert len(texts)==len(task_types)

    def __getitem__(self, item):
        return {
            'text':self.texts[item],
            'label':self.labels[item],
            'task':self.tasks[item],
            'task_type': self.task_types[item]
        }

    def __len__(self):
        return len(self.texts)

class SOCKETDataModule(LightningDataModule):
    def __init__(self,
                 model_name_or_path,
                 model_cache_dir=None,
                 tasks=None,
                 data_cache_dir=None,
                 train_batch_size=8,
                 eval_batch_size=16,
                 num_workers=4,
                 max_length=512,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_name_or_path','model_cache_dir'])
        self.list_of_tasks = tasks.split(',')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=model_cache_dir)
        self.task2idx = {task:i for i,task in enumerate(self.list_of_tasks)}

        self.datasets = {}
        self.dataset_info = {}

        input_datasets = {}
        splits = ['train','test','validation']
        for idx,task in enumerate(self.list_of_tasks):
            input_datasets[task]={}
            self.dataset_info[task]={}
            dataset = load_dataset(
                'Blablablab/SOCKET',task,
                cache_dir=self.hparams.data_cache_dir,
            )
            self.dataset_info[task]['task_idx']=idx

            # get the labels of each task as well as whether it is a classification or regression task
            labels = dataset['train']['label']
            label_type = dataset['train'].features['label']
            if type(label_type)==Value: # Regression
                self.dataset_info[task]['num_labels'] = 1
                self.dataset_info[task]['task_type'] = 'regression'
                self.dataset_info[task]['n_training_samples'] = len(labels)
            elif type(label_type) == ClassLabel: # Classification
                num_classes = label_type.num_classes
                self.dataset_info[task]['task_type']='classification'
                self.dataset_info[task]['num_labels']=num_classes
                self.dataset_info[task]['n_training_samples']=len(labels)
            elif type(label_type) == Sequence: # Span detection
                self.dataset_info[task]['task_type']='span'
                self.dataset_info[task]['num_labels']=3
                self.dataset_info[task]['n_training_samples']=len(labels)

            for split in splits:
                input_datasets[task][split]=dataset[split]


        merged_datasets = self.merge_datasets(input_datasets,splits)

        for split in splits:
            texts = merged_datasets[split]['texts']
            labels = merged_datasets[split]['labels']
            tasks = merged_datasets[split]['tasks']
            task_types = merged_datasets[split]['task_types']
            self.datasets[split] = DefaultDataset(
                texts=texts, labels=labels, tasks=tasks, task_types=task_types
            )

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SOCKETDataModule")
        parser.add_argument("--tasks", type=str, default=None)
        parser.add_argument("--data_cache_dir", type=str, default=None)
        parser.add_argument("--train_batch_size", type=int, default=8)
        parser.add_argument("--eval_batch_size", type=int, default=16)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--max_length", type=int, default=512)
        return parent_parser

    def setup(self, stage=None):
        return

    def merge_datasets(self,input_datasets,splits):
        outputs = {}
        for split in splits:
            total_samples=0
            outputs[split]={'texts':[],'labels':[],'tasks':[],'task_types':[]}
            for task,D in input_datasets.items():
                task_type = self.dataset_info[task]['task_type']
                task_idx = self.dataset_info[task]['task_idx'] # required to pass this to the dataloader as a tensor
                total_samples+=len(D[split])
                if task_type=='span':
                    texts,labels = self.process_span_dataset(D[split],task)
                else:
                    texts,labels = self.process_cls_dataset(D[split])

                outputs[split]['texts'].extend(texts)
                outputs[split]['labels'].extend(labels)
                outputs[split]['tasks'].extend([task_idx]*len(texts))
                outputs[split]['task_types'].extend([task_type]*len(texts))


            print(f'{total_samples} samples for {split}')
        return outputs

    def process_cls_dataset(self,dataset):
        all_texts,all_labels=[],[]
        for obj in dataset:
            text,label=obj['text'],obj['label']
            if np.isnan(label):
                continue
            else:
                all_texts.append(text)
                all_labels.append(label)
        return all_texts,all_labels


    def process_span_dataset(self,dataset,task):
        all_texts = []
        all_labels = []

        for ln,obj in enumerate(dataset):
            text = obj['text']
            text = text.replace('”', '"').replace('“', '"')
            # reduce length of string
            text = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.tokenize(text,max_length=self.hparams.max_length-2)
            )
            # all_texts.append(text)

            # 1) identify where span appears in text, and change it to [MASK]
            if task == 'toxic-span':
                label = obj['label']['toxic']
            elif task == 'propaganda-span':
                label = obj['label']['propaganda']
            if task == 'emotion-span':
                label = obj['label']['cause']
            # print(ln,'text:',text)
            # print(ln,'label:',label)
            tokenized_labels = {}
            for i, span in enumerate(label):
                span = span.replace('”', '"').replace('“', '"')
                span = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.tokenize(span, max_length=self.hparams.max_length-2))
                for c in ["'", '"', '.', '*', ',', '|', '(', ')', '?', '!', '[', ']']:
                    span = span.replace(c, '\\' + c)
                res = re.search(r'\b%s\b' % span, text)
                if res:
                    st, ed = res.span()
                else:
                    continue
                span = text[st:ed]
                tok_label = self.tokenizer.encode(span, add_special_tokens=False)
                if len(tok_label):
                    tokenized_labels[(st,ed)]=(span,tok_label)
                    text = text[:st] + self.tokenizer.mask_token + text[ed:]
            keys = sorted(tokenized_labels.keys())
            tokenized_labels2 = []
            for k in keys:
                v = tokenized_labels[k]
                tokenized_labels2.append((k[0], k[1], v[0], v[1]))

            tokenized_text = self.tokenizer.encode(text,add_special_tokens=False)

            tokenized_text_out = []
            labels = []
            n_masks = 0
            # print(ln,tokenized_labels2)
            # print(ln,tokenized_text)
            # print('\n\n')
            for i, idx in enumerate(tokenized_text):
                if idx == self.tokenizer.mask_token_id:
                    st, ed, span, tok_label = tokenized_labels2[n_masks]
                    tokenized_text_out.extend(tok_label)
                    labels.extend([2] + [1] * (len(tok_label) - 1))
                    n_masks += 1
                else:
                    tokenized_text_out.append(idx)
                    labels.append(0)  # O
            text_out = self.tokenizer.decode(tokenized_text_out)
            all_texts.append(text_out)
            all_labels.append(labels)
        return all_texts,all_labels

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], shuffle=True,
              num_workers=self.hparams.num_workers,
              collate_fn=self.collate_fn,
              batch_size=self.hparams.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.datasets['validation'], shuffle=False,
              num_workers=self.hparams.num_workers,
              collate_fn=self.collate_fn,
              batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], shuffle=False,
              num_workers=self.hparams.num_workers,
              collate_fn=self.collate_fn,
              batch_size=self.hparams.eval_batch_size)

    def predict_dataloader(self):
        return DataLoader(self.datasets['test'], shuffle=False,
              num_workers=self.hparams.num_workers,
              collate_fn=self.collate_fn,
              batch_size=self.hparams.eval_batch_size)

    def collate_fn(self, batch):
        sentences, labels = [],[]
        tasks, task_types = [],[]
        for obj in batch:
            text,label,task,task_type=obj['text'],obj['label'],obj['task'],obj['task_type']
            if self.tokenizer.sep_token in text:
                text=tuple(text.split(self.tokenizer.sep_token))
            sentences.append(text)
            tasks.append(task)
            if task_type=='span':
                labels.append(label)
            else:
                labels.append([label])

        # create output object
        output = self.tokenizer.batch_encode_plus(sentences,
               max_length=self.hparams.max_length, padding='longest',
               truncation='longest_first', return_length=True)
        max_length = len(output['input_ids'][0])
        output['tasks'] = tasks
        output['labels'] = []
        for label in labels:
            if len(label)>max_length:
                label=label[:max_length]
            elif len(label)<max_length:
                label=label+[0]*(max_length-len(label))
            output['labels'].append(label)

        for k,v in output.items():
            output[k]=torch.tensor(v)
        return output

if __name__=='__main__':
    import argparse

    # snippet to test if dataloader worksß
    list_of_tasks = [
        # 'hasbiasedimplication',
        # 'questionintimacy',
        # 'neutralizing-bias-pairs',
        'emotion-span',
        'propaganda-span',
    ]

    parser = argparse.ArgumentParser()
    parser = SOCKETDataModule.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    dict_args['tasks'] = ','.join(list_of_tasks)
    dict_args['train_batch_size'] = 8
    dict_args['num_workers']=1

    dm = SOCKETDataModule(
            model_name_or_path='bert-base-uncased',
            # model_cache_dir='/shared/3/projects/SOCKET/.cache/huggingface/transformers',
            **dict_args)
    dm.setup()
    for i,batch in enumerate(dm.train_dataloader()):
        continue

"""
python pl_data_loader.py --tasks=tweet_hate,tweet_irony
"""

