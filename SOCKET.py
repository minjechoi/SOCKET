# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The SOCKET Datasets"""


import datasets


_CITATION = """
@misc{choi2023llms,
      title={Do LLMs Understand Social Knowledge? Evaluating the Sociability of Large Language Models with SocKET Benchmark}, 
      author={Minje Choi and Jiaxin Pei and Sagar Kumar and Chang Shu and David Jurgens},
      year={2023},
      eprint={2305.14938},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
A unified evaluation benchmark dataset for evaludating socialbility of NLP models.
"""

_HOMEPAGE = "TBD"

_LICENSE = ""

#set up url or the file dir here
URL = "https://huggingface.co/datasets/Blablablab/SOCKET/blob/main/SOCKET_DATA/"

TASK_DICT = {
    'humor_sarcasm': [
        'hahackathon#humor_rating',
        'humor-pairs',
        'sarc',
        'tweet_irony',
        'hahackathon#is_humor',
    ],
    'offensive': [
        'contextual-abuse#IdentityDirectedAbuse',
        'contextual-abuse#PersonDirectedAbuse',
        'hahackathon#offense_rating',
        'hasbiasedimplication',
        'hateoffensive',
        'implicit-hate#explicit_hate',
        'implicit-hate#implicit_hate',
        'implicit-hate#incitement_hate',
        'implicit-hate#inferiority_hate',
        'implicit-hate#stereotypical_hate',
        'implicit-hate#threatening_hate',
        'implicit-hate#white_grievance_hate',
        'intentyn',
        'jigsaw#severe_toxic',
        'jigsaw#identity_hate',
        'jigsaw#threat',
        'jigsaw#obscene',
        'jigsaw#insult',
        'jigsaw#toxic',
        'offensiveyn',
        'sexyn',
        'talkdown-pairs',
        'toxic-span',
        'tweet_offensive'
    ],
    'sentiment_emotion': [
        'crowdflower',
        'dailydialog',
        'emobank#arousal',
        'emobank#dominance',
        'emobank#valence',
        'emotion-span',
        'empathy#distress',
        'empathy#distress_bin',
        'same-side-pairs',
        'sentitreebank',
        'tweet_emoji',
        'tweet_emotion',
        'tweet_sentiment'
    ],
    'social_factors': [
        'complaints',
        'empathy#empathy',
        'empathy#empathy_bin',
        'hayati_politeness',
        'questionintimacy',
        'stanfordpoliteness'
    ],
    'trustworthy': [
        'bragging#brag_achievement',
        'bragging#brag_action',
        'bragging#brag_possession',
        'bragging#brag_trait',
        'hypo-l',
        'neutralizing-bias-pairs',
        'propaganda-span',
        'rumor#rumor_bool',
        'two-to-lie#receiver_truth',
        'two-to-lie#sender_truth',
    ]
}

task2category = {}
for category, tasks in TASK_DICT.items():
    for task in tasks:
        task2category[task] = category

TASK_NAMES = []
for tasks in TASK_DICT.values():
    TASK_NAMES.extend(tasks)
TASK_NAMES = sorted(TASK_NAMES)

print(len(TASK_NAMES))
_URLs = {}
for task in TASK_NAMES:
    _URLs[task] = {}
    for s in ['train', 'test', 'val']:
        for t in ['text', 'labels']:
            task_url = '%s%s/%s_%s.txt'%(URL,task,s,t)
            task_url = task_url.replace('#','%23')
            _URLs[task][s + '_' + t] = task_url

class SOCKETConfig(datasets.BuilderConfig):
    def __init__(self, *args, type=None, sub_type=None, **kwargs):
        super().__init__(
            *args,
            name=f"{type}",
            **kwargs,
        )
        self.type = type
        self.sub_type = sub_type


class SOCKET(datasets.GeneratorBasedBuilder):
    """SOCKET Dataset."""

    BUILDER_CONFIGS = [
        SOCKETConfig(
            type=key,
            sub_type=None,
            version=datasets.Version("1.1.0"),
            description=f"This part of my dataset covers {key} part of SocialEval Dataset.",
        )
        for key in list(TASK_NAMES)
    ] 

    def _info(self):
        if self.config.type == "questionintimacy": 
            names = ['Very-intimate', 'Intimate', 'Somewhat-intimate', 'Not-very-intimate', 'Not-intimate', 'Not-intimate-at-all']
        elif self.config.type == "sexyn": 
            names = ['not sexism', 'sexism']
        elif self.config.type == "intentyn": 
            names = ['not intentional', 'intentional']
        elif self.config.type == "offensiveyn": 
            names = ['not offensive', 'offensive']
        elif self.config.type == "hasbiasedimplication": 
            names = ['not biased', 'biased']
        elif self.config.type == "trofi": 
            names = ['metaphor', 'non-metaphor']
        elif self.config.type == "sentitreebank": 
            names = ['positive', 'negative']
        elif self.config.type == "sarc": 
            names = ['sarcastic', 'literal']
        elif self.config.type == "stanfordpoliteness": 
            names = ['polite', 'impolite']
        elif self.config.type == "sarcasmghosh": 
            names = ['sarcastic', 'literal']
        elif self.config.type == "dailydialog": 
            names = ['noemotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
        elif self.config.type == "shortromance": 
            names = ['romantic', 'literal']
        elif self.config.type == "crowdflower": 
            names = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'love', 'fun', 'hate', 'happiness', 'relief', 'boredom', 'surprise', 'anger']
        elif self.config.type == "vua": 
            names = ['metaphor', 'non-metaphor']
        elif self.config.type == "shorthumor": 
            names = ['humorous', 'literal']
        elif self.config.type == "shortjokekaggle": 
            names = ['humorous', 'literal']
        elif self.config.type == "hateoffensive": 
            names = ['hate', 'offensive', 'neither']
        elif self.config.type == "emobank#valence": 
            names = ['valence(positive)']
        elif self.config.type == "emobank#arousal": 
            names = ['arousal(excited)']
        elif self.config.type == "emobank#dominance": 
            names = ['dominance(being_in_control)']
        elif self.config.type == "hayati_politeness": 
            names = ['impolite', 'polite']
        elif self.config.type == "jigsaw#toxic": 
            names = ['not toxic', 'toxic']
        elif self.config.type == "jigsaw#severe_toxic": 
            names = ['not severe toxic', 'severe toxic']
        elif self.config.type == "jigsaw#obscene": 
            names = ['not obscene', 'obscene']
        elif self.config.type == "jigsaw#threat": 
            names = ['not threat', 'threat']
        elif self.config.type == "jigsaw#insult": 
            names = ['not insult', 'insult']
        elif self.config.type == "jigsaw#identity_hate": 
            names = ['not identity hate', 'identity hate']
        elif self.config.type == "standup-comedy": 
            names = ['not funny', 'funny']
        elif self.config.type == "complaints": 
            names = ['not complaint', 'complaint']
        elif self.config.type == "hypo-l": 
            names = ['not hyperbole', 'hyperbole']
        elif self.config.type == "bragging#brag_action": 
            names = ['not action bragging', 'action bragging']
        elif self.config.type == "bragging#brag_feeling": 
            names = ['not feeling bragging', 'feeling bragging']
        elif self.config.type == "bragging#brag_achievement": 
            names = ['not achievement bragging', 'achievement bragging']
        elif self.config.type == "bragging#brag_possession": 
            names = ['not possession bragging', 'possession bragging']
        elif self.config.type == "bragging#brag_trait": 
            names = ['not trait bragging', 'trait bragging']
        elif self.config.type == "bragging#brag_affiliation": 
            names = ['not affiliation bragging', 'affiliation bragging']
        elif self.config.type == "contextual-abuse#IdentityDirectedAbuse": 
            names = ['not identity directed abuse', 'identity directed abuse']
        elif self.config.type == "contextual-abuse#AffiliationDirectedAbuse": 
            names = ['not affiliation directed abuse', 'affiliation directed abuse']
        elif self.config.type == "contextual-abuse#PersonDirectedAbuse": 
            names = ['not person directed abuse', 'person directed abuse']
        elif self.config.type == "contextual-abuse#CounterSpeech": 
            names = ['not counter speech', 'counter speech']
        elif self.config.type == "hahackathon#is_humor": 
            names = ['not humor', 'humor']
        elif self.config.type == "hahackathon#humor_rating": 
            names = ['humor rating']
        elif self.config.type == "hahackathon#offense_rating": 
            names = ['offense rating']
        elif self.config.type == "check_worthiness": 
            names = ['not check-worthy', 'check-worthy']
        elif self.config.type == "rumor#rumor_tf": 
            names = ['not rumor tf', 'rumor tf']
        elif self.config.type == "rumor#rumor_bool": 
            names = ['not rumor', 'rumor']
        elif self.config.type == "two-to-lie#deception": 
            names = ['not deception', 'deception']
        elif self.config.type == "two-to-lie#sender_truth": 
            names = ['lie', 'truth']
        elif self.config.type == "two-to-lie#receiver_truth": 
            names = ['lie', 'truth']
        elif self.config.type == "deceitful-reviews#true_rumor": 
            names = ['fake review', 'true review']
        elif self.config.type == "deceitful-reviews#positive": 
            names = ['negative', 'positive']
        elif self.config.type == "empathy#empathy": 
            names = ['empathy']
        elif self.config.type == "empathy#distress": 
            names = ['distress']
        elif self.config.type == "empathy#empathy_bin": 
            names = ['not empathy', 'empathy']
        elif self.config.type == "empathy#distress_bin": 
            names = ['not distress', 'distress bin']
        elif self.config.type == "implicit-hate#explicit_hate": 
            names = ['not explicit hate', 'explicit hate']
        elif self.config.type == "implicit-hate#implicit_hate": 
            names = ['not implicit hate', 'implicit hate']
        elif self.config.type == "implicit-hate#threatening_hate": 
            names = ['not threatening hate', 'threatening hate']
        elif self.config.type == "implicit-hate#irony_hate": 
            names = ['not irony hate', 'irony hate']
        elif self.config.type == "implicit-hate#other_hate": 
            names = ['not other hate', 'other hate']
        elif self.config.type == "implicit-hate#incitement_hate": 
            names = ['not incitement hate', 'incitement hate']
        elif self.config.type == "implicit-hate#inferiority_hate": 
            names = ['not inferiority hate', 'inferiority hate']
        elif self.config.type == "implicit-hate#stereotypical_hate": 
            names = ['not stereotypical hate', 'stereotypical hate']
        elif self.config.type == "implicit-hate#white_grievance_hate": 
            names = ['not white grievance hate', 'white grievance hate']
        elif self.config.type == "waseem_and_hovy#sexism": 
            names = ['not sexism', 'sexism']
        elif self.config.type == "waseem_and_hovy#racism": 
            names = ['not racism', 'racism']
        elif self.config.type == "humor-pairs": 
            names = ['the first sentence is funnier', 'the second sentence is funnier']
        elif self.config.type == "neutralizing-bias-pairs": 
            names = ['the first sentence is biased', 'the second sentence is biased']
        elif self.config.type == "same-side-pairs": 
            names = ['not same side', 'same side']
        elif self.config.type == "talkdown-pairs": 
            names = ['not condescension', 'condescension']
        elif self.config.type == "tweet_sentiment":
            names = ["negative", "neutral", "positive"]
        elif self.config.type == "tweet_offensive":
            names = ["not offensive", "offensive"]
        elif self.config.type == "tweet_irony":
            names = ["not irony", "irony"]
        elif self.config.type == "tweet_hate":
            names = ["not hate", "hate"]
        elif self.config.type == "tweet_emoji":
            names = [
                "❤",
                "😍",
                "😂",
                "💕",
                "🔥",
                "😊",
                "😎",
                "✨",
                "💙",
                "😘",
                "📷",
                "🇺🇸",
                "☀",
                "💜",
                "😉",
                "💯",
                "😁",
                "🎄",
                "📸",
                "😜",
            ]

        elif self.config.type == "tweet_emotion":
            names = ["anger", "joy", "optimism", "sadness"]
        elif self.config.type == "emotion-span": 
            names = ['cause']
            label_type = datasets.Sequence(feature={n:datasets.Value(dtype='string', id=None) for n in names})
            print(label_type)
        elif self.config.type == "propaganda-span": 
            names = ['propaganda']
            label_type = datasets.Sequence(feature={n:datasets.Value(dtype='string', id=None) for n in names})
        elif self.config.type == "toxic-span": 
            names = ['toxic']
            label_type = datasets.Sequence(feature={n:datasets.Value(dtype='string', id=None) for n in names})
            
        if self.config.type[-4:]=='span':
            label_type = label_type#datasets.Sequence(feature={n:datasets.Value(dtype='string') for n in names})
        elif len(names) > 1:
            label_type = datasets.features.ClassLabel(names=names)
        else:
            label_type = datasets.Value("float32")
        
    
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"text": datasets.Value("string"), 
                 "label": label_type}
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.type]
        data_dir = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"text_path": data_dir["train_text"], "labels_path": data_dir["train_labels"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"text_path": data_dir["test_text"], "labels_path": data_dir["test_labels"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"text_path": data_dir["val_text"], "labels_path": data_dir["val_labels"]},
            ),
        ]

    def _generate_examples(self, text_path, labels_path):
        """Yields examples."""

        with open(text_path, encoding="utf-8") as f:
            texts = f.readlines()
            print(len(texts))
        with open(labels_path, encoding="utf-8") as f:
            labels = f.readlines()
            print(len(labels))

        for i, text in enumerate(texts):
            yield i, {"text": text.strip(), "label": labels[i].strip() if self.config.type[-4:]!='span' else eval(labels[i])}