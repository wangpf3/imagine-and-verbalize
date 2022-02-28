import os
import json
import numpy as np
import pickle
import random
import math
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset


class Data_Helper(object):
    """docstring for Data_Helper"""
    def __init__(self, tokenizer, args, split='test'):
        super().__init__()

        self.tokenizer = tokenizer
        self.args = args

        self.data_dir = os.path.join('./data', args.dataset)

        cache_features_path = os.path.join(self.data_dir, 'features_{}_{}_enc{}_dec{}.pkl'.format(
                    tokenizer.name_or_path.replace('/', '-'), 
                    args.separation,
                    self.args.max_enc_length, 
                    self.args.max_dec_length)
        )

        from .data_processor_textualization import RawDataProcessor
        self.processor = RawDataProcessor(self.data_dir, tokenizer, args)

        if not args.do_train:
            self.testset = self.processor.load_datasplit(split)
        else:
            if not os.path.exists(cache_features_path):
                self.trainset = self.processor.load_trainset()
                self.devset = self.processor.load_devset()
                with open(cache_features_path, 'wb') as handle:
                    pickle.dump([self.trainset, self.devset], handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(cache_features_path, 'rb') as handle:
                    self.trainset, self.devset = pickle.load(handle)

    def data_collator(self, features):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        label_tensor = []

        for feature in features:
            max_num_mask = int(len(feature.entities) * self.args.mask_ratio)
            num_mask = random.choice(range(max_num_mask+1))
            input_entities = random.sample(feature.entities, len(feature.entities) - num_mask)
            if self.args.context or self.args.textualization:
                encoder_input_ids, encoder_attention_mask = self.processor._get_context_tensor(feature.context, input_entities)
            else:
                encoder_input_ids, encoder_attention_mask = self.processor._get_context_tensor('none', input_entities)

            encoder_input_tensor.append(encoder_input_ids)
            encoder_attention_mask_tensor.append(encoder_attention_mask)
            label_tensor.append(feature.label)

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, encoder_attention_mask_tensor, label_tensor])

    def sequential_iterate(self, dataset, batch_size):
        data_size = len(dataset)
        batch_num = math.ceil(data_size / batch_size)
                 
        for batch_id in range(batch_num):
            start_index = batch_id * batch_size
            end_index = min((batch_id+1) * batch_size, data_size)
            batch = dataset[start_index:end_index]
            yield batch

    def data_collator_for_inference(self, features):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []

        for feature in features:
            encoder_input_ids, encoder_attention_mask = self.processor._get_context_tensor(feature.context, feature.entities)

            encoder_input_tensor.append(encoder_input_ids)
            encoder_attention_mask_tensor.append(encoder_attention_mask)

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, encoder_attention_mask_tensor])

    def data_collator_for_inference_contextualized(self, features, batch_generated_context):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        for feature, generated_context in zip(features, batch_generated_context):
            encoder_input_ids, encoder_attention_mask = self.processor._get_context_tensor(generated_context, feature.entities)

            encoder_input_tensor.append(encoder_input_ids)
            encoder_attention_mask_tensor.append(encoder_attention_mask)

        return tuple(torch.tensor(t) for t in [encoder_input_tensor, 
                                            encoder_attention_mask_tensor])
