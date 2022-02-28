import os
import logging
from tqdm import tqdm, trange
import numpy as np
import math
import json

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

from .utils.data_helper import Data_Helper

class Build_GraphGenerator:
    def __init__(self, args):
        self.args = args

        ckpt_path = os.path.join(self.args.graph_generator_dir, 'model.ckpt.best')
        print('initialzing graph generator from', ckpt_path)
        save_ckpt = torch.load(ckpt_path, map_location='cpu')
        old_args = save_ckpt['args']
        self.args.model_type = old_args.model_type
        self.args.max_enc_length = old_args.max_enc_length
        self.args.max_dec_length = old_args.max_dec_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_type, cache_dir='../cache')

        self.tokenizer.add_tokens(['<SEP>'])
        with open(os.path.join(self.args.graph_generator_dir, 'relation_vocab.json'), 'rb') as handle:
            relation_dict = json.load(handle)
        for reltype in relation_dict:
            self.tokenizer.add_tokens(['<{}>'.format(reltype)])

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_type, cache_dir='../cache/')
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.load_state_dict(save_ckpt['ckpt'])
