# coding=UTF-8
import os
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import pickle
import numpy as np

from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from trainer import Trainer
from dataloader import DataLoader
from model import LMKE


def load_model():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--model_lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=64)  # 64
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=1e-7)

    parser.add_argument('--data', type=str, default='SZ-Taxi-2015_1_1_0_00')  # fb15k-237
    parser.add_argument('--plm', type=str, default='bert_tiny',
                        choices=['bert', 'bert_tiny', 'deberta', 'deberta_large', 'roberta', 'roberta_large'])  # bert
    parser.add_argument('--description', type=str, default='desc')

    parser.add_argument('--load_path', type=str, default="./params/SZ-Taxi-2015_1_1_0_00-bert_tiny-desc-batch_size=64-prefix_tuning=True-max_desc_length=256-epc_1_metric_fil_mrr.pt") # None
    parser.add_argument('--load_epoch', type=int, default=1)   # -1
    parser.add_argument('--load_metric', type=str, default='mrr')   # hits1

    parser.add_argument('--max_desc_length', type=int, default=256)  # 512

    # directly run test
    parser.add_argument('--link_prediction', default=True, action='store_true') # False
    parser.add_argument('--triple_classification', default=False, action='store_true')

    parser.add_argument('--add_tokens', default=True, action='store_true',
                        help='add entity and relation tokens into the vocabulary')  # False
    parser.add_argument('--p_tuning', default=True, action='store_true', help='add learnable soft prompts')  # False
    parser.add_argument('--prefix_tuning', default=True, action='store_true',
                        help='fix language models and only tune added components')  # False
    parser.add_argument('--rdrop', default=True, action='store_true')  # False
    parser.add_argument('--self_adversarial', default=True, action='store_true',
                        help='self adversarial negative sampling')  # False
    parser.add_argument('--no_use_lm', default=False, action='store_true')
    parser.add_argument('--use_structure', default=True, action='store_true')  # False
    parser.add_argument('--contrastive', default=True, action='store_true')  # False
    parser.add_argument('--wandb', default=False, action='store_true')

    parser.add_argument('--task', default='LP', choices=['LP', 'TC'])

    arg = parser.parse_args()

    if arg.task == 'TC':
        neg_rate = 1
    else:
        neg_rate = 0

    identifier = '{}-{}-{}-batch_size={}-prefix_tuning={}-max_desc_length={}'.format(arg.data, arg.plm, arg.description,
                                                                                     arg.batch_size, arg.prefix_tuning,
                                                                                     arg.max_desc_length)

    # Set random seed
    random.seed(arg.seed)
    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)

    device = torch.device('cuda')

    if arg.plm == 'bert':
        plm_name = "bert-base-uncased"
        t_model = 'bert'
    elif arg.plm == 'bert_tiny':
        plm_name = "prajjwal1/bert-tiny"
        t_model = 'bert'
    elif arg.plm == 'deberta':
        plm_name = 'microsoft/deberta-v3-base'
        t_model = 'bert'
    elif arg.plm == 'deberta_large':
        plm_name = 'microsoft/deberta-v3-large'
        t_model = 'bert'
    elif arg.plm == 'roberta_large':
        plm_name = "roberta-large"
        t_model = 'roberta'
    elif arg.plm == 'roberta':
        plm_name = "roberta-base"
        t_model = 'roberta'

    if arg.data == 'fb13':
        in_paths = {
            'dataset': arg.data,
            'train': './data/FB13/train.tsv',
            'valid': './data/FB13/dev.tsv',
            'test': './data/FB13/test.tsv',
            'text': ['./data/FB13/entity2text.txt', './data/FB13/relation2text.txt']
        }
    elif arg.data == 'umls':
        in_paths = {
            'dataset': arg.data,
            'train': './data/umls/train.tsv',
            'valid': './data/umls/dev.tsv',
            'test': './data/umls/test.tsv',
            'text': ['./data/umls/entity2textlong.txt', './data/umls/relation2text.txt']
        }
    elif arg.data == 'fb15k-237':
        in_paths = {
            'dataset': arg.data,
            'train': './data/fb15k-237/train.tsv',
            'valid': './data/fb15k-237/dev.tsv',
            'test': './data/fb15k-237/test.tsv',
            'text': ['./data/fb15k-237/FB15k_mid2description.txt',
                     # './data/fb15k-237/entity2textlong.txt',
                     './data/fb15k-237/relation2text.txt']
        }
    elif arg.data == 'wn18rr':
        in_paths = {
            'dataset': arg.data,
            'train': './data/WN18RR/train.tsv',
            'valid': './data/WN18RR/dev.tsv',
            'test': './data/WN18RR/test.tsv',
            'text': ['./data/WN18RR/my_entity2text.txt',
                     './data/WN18RR/relation2text.txt']
        }
    elif arg.data == 'SZ-Taxi-2015_1_1_0_00':
        in_paths = {
            'dataset': arg.data,
            'train': './data/SZ-Taxi/2015_1_1_0_00/train.tsv',
            'valid': './data/SZ-Taxi/2015_1_1_0_00/dev.tsv',
            'test': './data/SZ-Taxi/2015_1_1_0_00/dev.tsv',
            'text': ['./data/SZ-Taxi/2015_1_1_0_00/entity2text.txt',
                     './data/SZ-Taxi/2015_1_1_0_00/relation2text.txt']
        }
    # local
    model_path = "./cached_model/models--{}".format(plm_name)
    lm_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    lm_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    lm_model = AutoModel.from_pretrained(model_path, config=lm_config, local_files_only=True)

    data_loader = DataLoader(in_paths, lm_tokenizer, batch_size=arg.batch_size, neg_rate=neg_rate,
                             max_desc_length=arg.max_desc_length,
                             add_tokens=arg.add_tokens, p_tuning=arg.p_tuning, rdrop=arg.rdrop, model=t_model)

    if arg.add_tokens:
        data_loader.adding_tokens()
        lm_model.resize_token_embeddings(len(lm_tokenizer))

    model = LMKE(lm_model, n_ent=len(data_loader.ent2id), n_rel=len(data_loader.rel2id), add_tokens=arg.add_tokens,
                 contrastive=arg.contrastive)

    no_decay = ["bias", "LayerNorm.weight"]
    param_group = [
        {'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
                                        if ('lm_model' not in n) and
                                        (not any(nd in n for nd in no_decay))],
         'weight_decay': arg.weight_decay},
        {'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
                                        if ('lm_model' not in n) and
                                        (any(nd in n for nd in no_decay))],
         'weight_decay': 0.0},
    ]

    if not arg.prefix_tuning:
        param_group += [
            {'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
                                           if ('lm_model' in n) and
                                           (not any(nd in n for nd in no_decay))],  # name中不包含bias和LayerNorm.weight
             'weight_decay': arg.weight_decay},
            {'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
                                           if ('lm_model' in n) and
                                           (any(nd in n for nd in no_decay))],
             'weight_decay': 0.0},
        ]

    optimizer = AdamW(param_group)  # transformer AdamW

    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=data_loader.step_per_epc)

    checkpoint = torch.load(arg.load_path)
    # 仅加载模型的 state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    hyperparams = {
        'batch_size': arg.batch_size,
        'epoch': arg.epoch,
        'identifier': identifier,
        'load_path': arg.load_path,
        'evaluate_every': 1,
        'update_every': 1,
        'load_epoch': arg.load_epoch,
        'load_metric': arg.load_metric,
        'prefix_tuning': arg.prefix_tuning,
        'plm': arg.plm,
        'description': arg.description,
        'neg_rate': neg_rate,
        'add_tokens': arg.add_tokens,
        'max_desc_length': arg.max_desc_length,
        'p_tuning': arg.p_tuning,
        'rdrop': arg.rdrop,
        'use_structure': arg.use_structure,
        'self_adversarial': arg.self_adversarial,
        'no_use_lm': arg.no_use_lm,
        'contrastive': arg.contrastive,
        'task': arg.task,
        'wandb': arg.wandb
    }
    trainer = Trainer(data_loader, model, lm_tokenizer, optimizer, scheduler, device, hyperparams)
    if arg.link_prediction:
        trainer.link_prediction(split='test')
    elif arg.triple_classification:
        trainer.triple_classification(split='test')
    else:
        trainer.run()
    return model


if __name__ == '__main__':
    model = load_model()
    state_dict = model.state_dict()
    for _name, _param in state_dict.items():
        if (_name == 'ent_embeddings.weight' or _name == 'rel_embeddings.weight'):
            print("{}, {}, {}".format(_name, _param.shape, str(_param)))
        if (_name == 'ent_embeddings_transe.weight' or _name == 'rel_embeddings_transe.weight'):
            print("{}, {}, {}".format(_name, _param.shape, str(_param)))
    # model.eval()

# ent_embeddings.weight, torch.Size([166, 128]), tensor([[-0.1354, -0.7258,  0.4161,  ...,  0.9471, -1.1090,  1.9841],
#         [ 0.7129, -3.1560, -0.4422,  ...,  0.1201, -1.0821, -0.5624],
#         [ 0.9593,  0.1387,  1.2011,  ...,  0.8016, -1.0528,  1.3924],
#         ...,
#         [-1.0565,  0.6994, -2.0656,  ...,  0.2350,  0.2207, -0.0052],
#         [ 0.3089, -1.0730,  0.9932,  ...,  1.1208,  0.0562,  1.6061],
#         [ 1.7325,  0.6219, -0.2147,  ...,  0.5043,  1.7923,  2.2030]],
#        device='cuda:0')
# rel_embeddings.weight, torch.Size([26, 128]), tensor([[-1.7664,  0.2084,  1.2308,  ...,  1.6275, -0.9383,  0.4529],
#         [ 1.2997, -1.1001,  0.3138,  ..., -0.4696, -2.0378,  0.3462],
#         [ 0.0994, -0.4500,  2.0665,  ..., -1.8458,  0.6344,  1.3451],
#         ...,
#         [ 0.9745,  0.5027,  0.8639,  ..., -1.3735,  0.8471, -0.3733],
#         [-0.7106, -0.5693,  0.6732,  ..., -1.0982, -2.7803, -0.1623],
#         [ 0.2831, -0.9801, -0.4978,  ..., -0.2033,  0.2786,  0.1387]],