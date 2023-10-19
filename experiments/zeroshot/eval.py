import os
import re
import ast
import string
from collections import Counter
import argparse

import emoji
import numpy as np
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

## functions for span detection task
def find_substring_indices(s, substrings):
    # get all character indices in string based on substrings
    indices = []
    for substring in substrings:
        start = 0
        while start < len(s):
            start = s.find(substring, start)
            if start == -1:
                break
            indices.extend(list(range(start,start + len(substring))))
            start += len(substring)
    return sorted(indices)

def get_span_f1(predictions, gold):
    """
    Based on Jaccard similarity
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1 if len(predictions)==0 else 0
    nom = 2*len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions))+len(set(gold))
    return nom/denom

def extract_spans(text):
    # extract spans based on two rules: (1) quoted substrings, or (2) original string
    quoted = re.findall(r'"(.*?)"', text)
    if len(quoted):
        return quoted
    else:
        return [text]
    
def longest_common_substring(S1, S2):
    # longest common substring between S1 (input text) vs S2 (substring)
    m = [[0] * (1 + len(S2)) for _ in range(1 + len(S1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(S1)):
        for y in range(1, 1 + len(S2)):
            if S1[x - 1] == S2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return S1[x_longest - longest: x_longest]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='results/')
    args = parser.parse_args()

    data_dir = args.data_dir
        
    # load prompt data
    ppt_df = pd.read_csv('socket_prompts.csv')

    # load files from all model settings
    files = sorted(os.listdir(data_dir))
    
    out = []
    for file in files:
        model = file.split('.')[0]
        df=pd.read_csv(os.path.join(data_dir,file),sep='\t')

        for task in sorted(df.task.unique()):
            df2=df[df.task==task]
            print(task)
            info = ppt_df[ppt_df.task==task].values[0]
            task_type = info[0]
            preds = []
            if task_type=='REG':
                # regression
                for line in df2.generated_text:
                    if type(line)==str:
                        scores = re.findall(r'[0-9](?:\.[0-9])?',line)
                        if len(scores)==1:
                            score = float(scores[0])
                        else:
                            score = None
                    else:
                        score = None
                    preds.append(score)
                    
                # percentage of answers found
                cn = Counter(preds)
                n_miss = cn[None]
                
                # fill missing answers
                if task in ['hahackathon#humor_rating','hahackathon#offense_rating',
                            'emobank#arousal','emobank#dominance','emobank#valence']:
                    ceil = 5.0
                elif task in ['empathy#distress','empathy#empathy']:
                    ceil = 7.0
                df2['pred'] = preds
                df2['pred']=df2['pred'].apply(lambda x: np.random.uniform(0, ceil) if pd.isnull(x) else x)
                df2[['label','pred']]=df2[['label','pred']].astype(float)
                
                # compute performance
                corr = df2['pred'].astype(float).corr(df2['label'].astype(float))
                out.append((model,task,task_type,'hit_rate',1-n_miss/len(preds)))
                out.append((model,task,task_type,'corr_original',corr))
                out.append((model,task,task_type,'corr_score',(corr+1)/2))
                    
            elif task_type in ['PAIR','CLS']:
                # classification
                options = ast.literal_eval(info[-1].lower())
                for line in df2.generated_text:
                    if task=='tweet_emoji':
                        if type(line)==str:
                            emojis= list(set([c for c in line if c in emoji.UNICODE_EMOJI['en']]))
                            if len(emojis)==1:
                                pred = emojis[0]
                                if pred in options:
                                    pred = options.index(pred)
                                    preds.append(pred)
                                else:
                                    preds.append(None)
                            else:
                                preds.append(None)
                        else:
                            preds.append(None)
                    else:
                        # for other tasks
                        if type(line)==str:
                            line = line.lower().strip()
                            flag = False
                            pred = None
                            for idx,opt in enumerate(options):
                                if flag==True:
                                    break
                                if line.startswith(opt):
                                    flag=True
                                    pred = idx
                                    break
                            if flag==True:
                                preds.append(idx)
                            else:
                                preds.append(None)
                        else:
                            preds.append(None)
                                
                # percentage of answers found
                cn = Counter(preds)
                n_miss = cn[None]

                # fill missing values with random numbers 
                df2['pred'] = preds
                df2['pred']=df2['pred'].apply(lambda x: np.random.randint(0, len(options)) if pd.isnull(x) else x)
                df2[['label','pred']]=df2[['label','pred']].astype(float)
                df2[['label','pred']]=df2[['label','pred']].astype(int)
                
                # get scores
                precision, recall, f1, _ =  precision_recall_fscore_support(df2['label'], df2['pred'], average='macro')
                out.append((model,task,task_type,'hit_rate',1-n_miss/len(preds)))
                out.append((model,task,task_type,'f1',f1))                
                    
            elif task_type=='SPAN':
                scores = []
                n_miss = 0
                for text,label,pred in df2[['text','label','generated_text']].values:
                    label = ast.literal_eval(label)
                    spans = list(label.values())[0]
                    if type(pred)==str:
                        pred = extract_spans(pred)
                        pred2 = [longest_common_substring(text,span) for span in pred]
                        pred2 = [x for x in pred2 if len(x)>=3]
                        pred_indices = find_substring_indices(text, pred2)
                    else:
                        pred_indices = []
                        n_miss += 1
                    true_indices = find_substring_indices(text, spans)
                    f1 = get_span_f1(pred_indices, true_indices)
                    scores.append(f1)
                out.append((model,task,task_type,'hit_rate',1-n_miss/len(df2)))
                out.append((model,task,task_type,'f1',f1))
    
    # save results
    df_out = pd.DataFrame(out,columns=['model','task','task_type','metric','score'])
    df_out.to_csv('all_scores.df.tsv',sep='\t',index=False)
