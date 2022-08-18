import torch, pickle, argparse, random, os, time, pdb, json, copy
import numpy as np
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from torch.autograd import Variable
# from utils_info import *
use_cuda = torch.cuda.is_available()

etype_map = {'台风': 0, '需求增加': 1, '市场价格下降': 2, '寒潮': 3, '市场价格提升': 4, '其他自然灾害': 5, '供给减少': 6,
              '供给增加': 7, '销量（消费）减少': 8, '需求减少': 9, '进口下降': 10, '洪涝': 11, '其他贸易摩擦': 12, '负向影响': 13,
              '猪瘟': 14, '销量（消费）增加': 15, '限产': 16, '运营成本提升': 17, '其他畜牧疫情': 18, '正向影响': 19, '干旱': 20,
              '运营成本下降': 21, '出口下降': 22, '霜冻': 23, '其他或不明确': 24, '进口增加': 25, '禽流感': 26, '地震': 27,
              '对华反倾销': 28, '出口增加': 29, '对华加征关税': 30, '产品利润下降': 31, '产品利润增加': 32, '猪口蹄疫': 33,
              '对他国反倾销': 34, '滞销': 35, '牛口蹄疫': 36, '山洪': 37, '冰雹': 38}

etype_id2type = {etype_map[iid]:iid for iid in etype_map}

tt_map = {'Rea2Rea-product-H': 0, 'Rea2Rea-product-T': 1, 'Rea2Rea-region-H':  2, 'Rea2Rea-region-T':  3,
          'Rea2Rea-industry-H': 4, 'Rea2Rea-industry-T': 5, 'Rea2Res-product-H': 6, 'Rea2Res-product-T': 7,
          'Rea2Res-region-H': 8, 'Rea2Res-region-T': 9,  'Rea2Res-industry-H': 10, 'Rea2Res-industry-T': 11,
          'Res2Res-product-H': 12, 'Res2Res-product-T': 13, 'Res2Res-region-H': 14, 'Res2Res-region-T': 15,
          'Res2Res-industry-H': 16, 'Res2Res-industry-T': 17,  'Res2Rea-product-H': 18, 'Res2Rea-product-T': 19,
          'Res2Rea-region-H':  20, 'Res2Rea-region-T': 21,  'Res2Rea-industry-H':  22, 'Res2Rea-industry-T': 23}
'''
Rea2Rea/Rea2Res refer to the Intra/Inter-field in the cause table, 
while Res2Res/Res2Rea refer to the Intra/Inter-field in the effect table.
'''

tt_id2type = {tt_map[iid]:iid for iid in tt_map}



max_seq_len = 150 + 1


def load_ids(filename):
    with open(filename, "r", encoding='utf-8') as f:
        lines = f.readlines()
    ids = [iid.strip() for iid in lines]
    return ids


def load_gold_answer(data_path):
    dev_answer_list, test_answer_list, train_answer_list = [], [], []

    with open(os.path.join(data_path, "raw_data.txt"), "r", encoding='utf-8') as f:
        lines = f.readlines()

    dev_ids = load_ids(os.path.join(data_path, 'split_data/dev_ids.txt'))
    test_ids = load_ids(os.path.join(data_path, 'split_data/test_ids.txt'))
    train_ids = load_ids(os.path.join(data_path, 'split_data/train_ids.txt'))
    
    for line in lines:
        line = eval(line)
        if line['text_id'] in dev_ids:
            dev_answer_list.append({'text_id': line['text_id'], 'result': line['result']})
        elif line['text_id'] in test_ids:
            test_answer_list.append({'text_id': line['text_id'], 'result': line['result']})
        elif line['text_id'] in train_ids:
            train_answer_list.append( {'text_id': line['text_id'], 'result': line['result']}  )
    return dev_answer_list, test_answer_list, train_answer_list



def Mkdir(path): 
    if not (os.path.isdir(path)):
        os.makedirs(path)


def write_predicted_results(task_name, name, epoch, results):
    with open("../res/{}/{}_{}.txt".format(task_name, name, str(epoch)), "w", encoding='utf-8') as f:
        f.write("\n".join(results))

def get_optimizer(model, args, warmup_steps, num_training_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    return optimizer, scheduler

def trans_to_cuda(variable):
    if use_cuda:
        return variable.cuda()
    else:
        return variable

def load_data(filename, debug):
    with open(filename, "rb") as f:
        lines = pickle.load(f)
        f.close()
    if debug:
        lines = lines[:200]
    return lines



def mask_select(inputs, mask, shape):
    return torch.masked_select(inputs, mask).reshape(shape)
