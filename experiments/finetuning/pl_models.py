import faulthandler
faulthandler.enable()

import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.metrics import accuracy_score,f1_score
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from pytorch_lightning import Trainer
from transformers import (
    AutoConfig,AutoModel,
    get_linear_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup
)

from pl_data_loader import SOCKETDataModule


class SOCKETModule(LightningModule):
    def __init__(
        self,
        list_of_tasks,
        dataset_info,
        model_name_or_path,
        model_cache_dir=None,
        learning_rate=1e-5,
        hidden_size=768,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        warmup_steps=0.0,
        dropout_prob=0.5,
        freeze_encoder=False,
        **kwargs,

    ):
        super().__init__()
        self.save_hyperparameters()

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SOCKETModule")
        parser.add_argument("--model_name_or_path", type=str, default=None)
        parser.add_argument("--model_cache_dir", type=str,
                            default='../../.cache/')
        parser.add_argument("--hidden_size", type=int, default=768)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        parser.add_argument("--warmup_steps", type=float, default=0.0)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--dropout_prob", type=float, default=0.5)
        parser.add_argument("--freeze_encoder", action='store_true')
        return parent_parser

    def setup(self, stage: str = None) -> None:
        # load model
        self.config = AutoConfig.from_pretrained(self.hparams.model_name_or_path,
                            cache_dir=self.hparams.model_cache_dir)
        self.encoder = AutoModel.from_pretrained(self.hparams.model_name_or_path,
                config=self.config, cache_dir=self.hparams.model_cache_dir)

        # create task-specific heads based on the mappings
        self.dataset_info = self.hparams.dataset_info
        self.list_of_tasks = self.hparams.list_of_tasks
        classifier_dict = {}
        for task in self.list_of_tasks:
            info = self.dataset_info[task]
            task_type=info['task_type']
            if task_type=='span':
                head = SpanHead(
                    num_labels=3,
                    hidden_size=self.hparams.hidden_size,
                    dropout_prob=self.hparams.dropout_prob)
            else:
                head = ClassifierHead(
                    num_labels=info['num_labels'],
                    hidden_size=self.hparams.hidden_size,
                    dropout_prob=self.hparams.dropout_prob)
            classifier_dict[task] = head
        self.classifier_dict = nn.ModuleDict(classifier_dict)

        if stage!='fit':
            return

        train_loader = self.trainer.datamodule.train_dataloader()
        tb_size = self.trainer.datamodule.hparams.train_batch_size
        # Calculate total steps
        if self.trainer.max_epochs>0:
            self.total_steps= len(train_loader) * self.trainer.max_epochs
        else:
            self.total_steps = self.trainer.max_steps

        if self.hparams.warmup_steps<=1:
            self.hparams.warmup_steps = int(self.total_steps*self.hparams.warmup_steps)
        else:
            self.hparams.warmup_steps = int(self.hparams.warmup_steps)

        print('Max steps:',self.total_steps)
        print("tb size",tb_size)
        print("Len train loader",len(train_loader))
        print("Max epochs",self.trainer.max_epochs)
        print('Warmup steps:',self.hparams.warmup_steps)

    def forward(self, batch):
        # get outputs from encoder
        if self.config.model_type in ['bloom','distilbert']:
            outputs = self.encoder(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                )
        else:
            outputs = self.encoder(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'] if 'token_type_ids' in batch else None,
                )
        return outputs[0]

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)

        # organize by tasks
        tasks = [self.list_of_tasks[i] for i in batch['tasks'].detach().cpu().tolist()]
        unique_tasks = list(set(tasks))

        total_loss = 0

        for task_no,task in enumerate(unique_tasks):
            # get indices from the list of tasks
            task_type = self.dataset_info[task]['task_type']
            idxs = [i for i,x in enumerate(tasks) if x==task]

            # if span
            if self.dataset_info[task]['task_type'] == 'span':
                if task_type == 'span':
                    task_outputs = torch.stack([output[i] for i in idxs], 0)
                    labels = torch.stack([batch['labels'][i] for i in idxs], 0)
                    attention_mask = torch.stack([batch['attention_mask'][i] for i in idxs]).bool()
                    logits = self.classifier_dict[task](task_outputs)
                    labels = labels[attention_mask].long()
                    logits = logits[attention_mask]
                    loss = F.cross_entropy(logits, labels)
            else:
                pooled_output = output[:,0] # use [CLS] token
                task_outputs = torch.stack([pooled_output[i] for i in idxs], 0)
                logits = self.classifier_dict[task](task_outputs)
                labels = torch.stack([batch['labels'][i][0] for i in idxs])
                # regression
                if task_type == 'regression':
                    logits = logits.reshape(-1)
                    labels = labels.float().reshape(-1)
                    loss = F.mse_loss(logits, labels)
                # classification
                elif task_type == 'classification':
                    labels = labels.long()
                    loss = F.cross_entropy(logits, labels)

            self.log(f"train_loss_{task}", loss,
                     on_step=True, on_epoch=True, prog_bar=False, logger=True)
            if task_no==0:
                total_loss = loss
            else:
                total_loss += loss

        self.log(f"train_loss_total", total_loss,
                 on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)

        # organize by tasks
        task_specific_values = {}
        assert len(batch['tasks'])==len(batch['input_ids'])
        tasks = [self.list_of_tasks[i] for i in batch['tasks'].detach().cpu().tolist()]
        unique_tasks = list(set(tasks))
        # print(unique_tasks)

        for task_no,task in enumerate(unique_tasks):
            # get indices from the list of tasks
            idxs = [i for i,x in enumerate(tasks) if x==task]
            task_type = self.dataset_info[task]['task_type']

            # if span
            if task_type == 'span':
                task_outputs = torch.stack([output[i] for i in idxs], 0)
                labels = torch.stack([batch['labels'][i] for i in idxs],0).long()
                attention_mask = torch.stack([batch['attention_mask'][i] for i in idxs]).bool()
                logits = self.classifier_dict[task](task_outputs)
                losses = F.cross_entropy(
                    logits[attention_mask],
                    labels[attention_mask],
                    reduction='none').detach().cpu().tolist()
                # print(task_type,logits.size(),labels.size())
                f1_values=[]
                for i in range(len(labels)):
                    attn=attention_mask[i]
                    label=labels[i][attn]
                    logit=logits[i][attn]
                    y_true=(label>0).long().detach().cpu().tolist()
                    y_pred=(logit[:,0]<0.5).long().detach().cpu().tolist()
                    if sum(y_true)>0:
                        f1=f1_score(y_true,y_pred)
                        f1_values.append(f1)
                task_specific_values[task] = {
                    'f1_scores':f1_values,
                    'losses': losses}
            else:
                pooled_output = output[:,0] # use [CLS] token
                # print(batch['input_ids'].size(), len(idxs), pooled_output.size())
                task_outputs = torch.stack([pooled_output[i] for i in idxs], 0)
                logits = self.classifier_dict[task](task_outputs)
                labels = torch.stack([batch['labels'][i][0] for i in idxs])

                if task_type == 'regression':
                    # regression
                    logits = logits.reshape(-1)
                    labels = labels.float().reshape(-1)
                    losses = F.mse_loss(logits,
                                        labels, reduction='none').detach().cpu().tolist()
                    labels = labels.float()
                    task_specific_values[task] = {
                        'predictions': logits.detach().cpu().tolist(),
                        'answers': labels.detach().cpu().tolist(),
                        'losses': losses}

                elif task_type == 'classification':
                    # classification
                    labels = labels.long()
                    losses = F.cross_entropy(logits, labels,
                                             reduction='none').detach().cpu().tolist()
                    task_specific_values[task] = {
                        'predictions': logits.argmax(1).detach().cpu().tolist(),
                        'answers': labels.detach().cpu().tolist(),
                        'losses': losses}

        return task_specific_values

    def validation_epoch_end(self, outputs):
        total_losses = []
        task_specific_values={}
        for i,output in enumerate(outputs):
            for task,obj2 in output.items():
                if task not in task_specific_values:
                    task_specific_values[task]={k:v for k,v in obj2.items()}
                else:
                    for k,v in obj2.items():
                        task_specific_values[task][k].extend(v)
                total_losses.extend(obj2['losses'])

        log_dict = {}
        log_dict['val_total_loss'] = np.mean(total_losses)
        for task,obj in task_specific_values.items():
            task_type = self.dataset_info[task]['task_type']
            # print(obj)
            if task_type=='classification':
                acc=accuracy_score(y_true=obj['answers'],y_pred=obj['predictions'])
                log_dict[f'val_{task}_acc']=round(acc,3)
                f1 = f1_score(y_true=obj['answers'], y_pred=obj['predictions'], average='macro')
                log_dict[f'val_{task}_f1'] = round(f1, 3)
            elif task_type=='span':
                f1 = np.mean(obj['f1_scores'])
                log_dict[f'val_{task}_f1'] = round(f1, 3)

            log_dict[f'val_{task}_loss'] = np.mean(obj['losses'])
        # log_dict['val_total_acc'] = np.mean([v for k,v in log_dict.items() if k.endswith('_acc')])
        print(log_dict)
        self.log_dict(log_dict,prog_bar=False)
        return

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        total_losses = []
        task_specific_values={}
        for i,output in enumerate(outputs):
            for task,obj2 in output.items():
                if task not in task_specific_values:
                    task_specific_values[task]={k:v for k,v in obj2.items()}
                else:
                    for k,v in obj2.items():
                        task_specific_values[task][k].extend(v)
                total_losses.extend(obj2['losses'])

        log_dict = {}
        log_dict['test_total_loss'] = np.mean(total_losses)
        for task,obj in task_specific_values.items():
            task_type = self.dataset_info[task]['task_type']
            # print(obj)
            if task_type=='classification':
                acc=accuracy_score(y_true=obj['answers'],y_pred=obj['predictions'])
                log_dict[f'test_{task}_acc']=round(acc,3)
                f1 = f1_score(y_true=obj['answers'], y_pred=obj['predictions'], average='macro')
                log_dict[f'test_{task}_f1'] = round(f1, 3)
            elif task_type=='span':
                f1 = np.mean(obj['f1_scores'])
                log_dict[f'test_{task}_f1'] = round(f1, 3)
            elif task_type=='regression':
                corr,_ = pearsonr(obj['answers'],obj['predictions'])
                log_dict[f'test_{task}_corr'] = round(corr, 3)

            log_dict[f'test_{task}_loss'] = np.mean(obj['losses'])
        # log_dict['val_total_acc'] = np.mean([v for k,v in log_dict.items() if k.endswith('_acc')])
        print(log_dict)
        self.log_dict(log_dict,prog_bar=False)
        return log_dict

    def predict_step(self, batch, batch_idx):
        result = {}
        pooled_output = self.forward(batch)
        # instead of matching each sample to its corresponding task's head, we get the values for everything
        for model_task,clf in self.classifier_dict.items():
            logits = clf(pooled_output)
            if self.dataset_info[model_task]['task_type']=='regression':
                logits = logits.reshape(-1,1)
            elif self.dataset_info[model_task]['task_type'] == 'classification':
                logits = logits.softmax(dim=1)
            logits = logits.detach().cpu().tolist()
            result[model_task]=logits
        tasks = batch['tasks'].detach().cpu().tolist()
        return (result,tasks) # dictionary containing which models were used

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        no_decay_params = []
        decay_params = []

        all_params = []
        if self.hparams.freeze_encoder==False:
            all_params.append(self.encoder)
        all_params.extend(list(self.classifier_dict.values()))
        for module in all_params:
            no_decay_params.extend([p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)])
            decay_params.extend([p for n, p in module.named_parameters() if any(nd in n for nd in no_decay)])

        optimizer_grouped_parameters = [
            {"params": no_decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": decay_params, "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        if self.hparams.warmup_steps>0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.total_steps
            )
        else:
            scheduler = get_constant_schedule(
                optimizer
            )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        self.optimizer = optimizer
        return [optimizer], [scheduler]

# Classifier head for generic task
class ClassifierHead(nn.Module):
    def __init__(self, num_labels=2, hidden_size=768, dropout_prob=0.1):
        super().__init__()
        # from transformers.models.roberta.modeling_roberta import *
        self.dense = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs):
        # note that the inputs here are the outputs that have already been grouped by task
        x = self.dropout(inputs)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits

class SpanHead(nn.Module):
    def __init__(self, num_labels=3, hidden_size=768, dropout_prob=0.1):
        super().__init__()
        self.seq = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1,
                         bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size*2, num_labels)

    def forward(self, inputs):
        # note that the inputs here are the outputs that have already been grouped by task
        x = self.seq(inputs)[0]
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


if __name__=='__main__':
    import argparse
    # snippet to test if dataloader works
    list_of_tasks = [
        'emotion-span',
        'propaganda-span'
    ]

    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = SOCKETDataModule.add_model_specific_args(parser)
    parser = SOCKETModule.add_model_specific_args(parser)
    args = parser.parse_args()
    args.train_batch_size=8
    args.accelerator='gpu'
    args.devices=1
    args.warmup_steps=0.06
    args.max_epochs=3
    args.num_workers=1
    args.model_name_or_path='microsoft/deberta-v3-base'
    dict_args = vars(args)
    dict_args['tasks'] = ','.join(list_of_tasks)
    print(dict_args)

    dm = SOCKETDataModule(
            **dict_args)

    model = SOCKETModule(list_of_tasks=list_of_tasks, dataset_info=dm.dataset_info,**dict_args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model,datamodule=dm)
