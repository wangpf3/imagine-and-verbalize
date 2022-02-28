import os
import json
import math
import numpy as np
import pickle
import random
from tqdm import tqdm
import copy

import torch
from torch.utils.data import TensorDataset
from .data_processor import RawDataProcessor


class Data_Helper(object):
    """docstring for Data_Helper"""
    def __init__(self, tokenizer, args, split='test', inference=False):
        super().__init__()

        self.tokenizer = tokenizer
        self.args = args

        self.data_dir = os.path.join('./data', args.dataset)

        self.data_collator = self.data_collator_for_concept2sentence
        self.processor = RawDataProcessor(self.data_dir, tokenizer, args)

        if inference:
            self.testset = self.processor.load_datasplit(split)
        else:
            self.trainset = self.processor.load_trainset()
            self.devset = self.processor.load_devset()

            if args.graph_source_alpha > 0:
                self.trainset_with_groundtruth = self.processor.load_datasplit('train.groundtruth')

    def sequential_iterate(self, dataset, batch_size):
        data_size = len(dataset)
        batch_num = math.ceil(data_size / batch_size)
                 
        for batch_id in range(batch_num):
            start_index = batch_id * batch_size
            end_index = min((batch_id+1) * batch_size, data_size)
            batch = dataset[start_index:end_index]
            yield batch

    def data_collator_for_concept2sentence_inference(self, features):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        for feature in features:
            input_seq = self.processor.format_input(feature.context, feature.entities, feature.relations)
            encoder_input = self.tokenizer(input_seq, padding='max_length', max_length=self.args.max_enc_length, truncation=True)

            encoder_input_tensor.append(encoder_input['input_ids'])
            encoder_attention_mask_tensor.append(encoder_input['attention_mask'])

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, 
                                            encoder_attention_mask_tensor])

    def data_collator_for_graph2story_inference(self, features, batch_generated_context, batch_generated_graph):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        for feature, generated_context, generated_graph in zip(features, batch_generated_context, batch_generated_graph):
            input_seq = 'context: ' + generated_context + ' entities: ' +  '<SEP>'.join(feature.entities) + ' relations: ' + generated_graph
            encoder_input = self.tokenizer(input_seq, padding='max_length', max_length=self.args.max_enc_length, truncation=True)

            encoder_input_tensor.append(encoder_input['input_ids'])
            encoder_attention_mask_tensor.append(encoder_input['attention_mask'])

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, 
                                            encoder_attention_mask_tensor])

    def data_collator_for_concept2graph_inference(self, tokenizer, features, batch_generated_context):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        for feature, generated_context in zip(features, batch_generated_context):
            input_seq = 'context: ' + generated_context + ' entities: ' +  '<SEP>'.join(feature.entities)
            encoder_input = tokenizer(input_seq, padding='max_length', max_length=self.args.max_enc_length, truncation=True)

            encoder_input_tensor.append(encoder_input['input_ids'])
            encoder_attention_mask_tensor.append(encoder_input['attention_mask'])

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, 
                                            encoder_attention_mask_tensor])

    def data_collator_for_concept2sentence(self, features):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        text_label_tensor = []

        for feature in features:
            input_entities = random.sample(feature.entities, len(feature.entities))
            input_seq = self.processor.format_input(feature.context, input_entities, feature.relations)
            encoder_input = self.tokenizer(input_seq, padding='max_length', max_length=self.args.max_enc_length, truncation=True)

            encoder_input_tensor.append(encoder_input['input_ids'])
            encoder_attention_mask_tensor.append(encoder_input['attention_mask'])

            text_label_tensor.append(feature.text_label)

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, 
                                            encoder_attention_mask_tensor,
                                            text_label_tensor])
