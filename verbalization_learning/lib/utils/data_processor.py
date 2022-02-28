import json
import os 
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional
import random

import torch
from torch.utils.data import Dataset, TensorDataset

@dataclass(frozen=True)
class InputExample:

    context: str
    entities: List[str]
    relations: List[List[str]]
    text_label: List[int]

class TrainingDataset(Dataset):

    features: List[InputExample]

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputExample:
        return self.features[i]

class RawDataProcessor:
    def __init__(self, data_dir, tokenizer, args):

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.args = args
        self.train_sample_ids = None

    def load_trainset(self,):
        return self._get_raw_dataset(os.path.join(self.data_dir, "train.json"))

    def load_devset(self,):
        return self._get_tensor_dataset(os.path.join(self.data_dir, "dev.json"))

    def load_testset(self,):
        return self._get_tensor_dataset(os.path.join(self.data_dir, "test.json"))

    def load_datasplit(self, split):
        # return self._get_tensor_dataset_for_inference(os.path.join(self.data_dir, "{}.json".format(split)))
        return self._get_raw_dataset(os.path.join(self.data_dir, "{}.json".format(split)))

    def _get_tensor_graph_input(self, relations):

        sequence = ''
        if len(relations) == 0:
            return sequence
        triplet = relations[0]
        sequence += triplet[0] + '<{}>'.format(triplet[1]) + triplet[2]

        for triplet in relations[1:]:
            sequence += '<SEP>' + triplet[0] + '<{}>'.format(triplet[1]) + triplet[2]
        return sequence

    def _get_tensor_text_label(self, raw_label):
        label_ids = self.tokenizer.encode(raw_label, add_special_tokens=False)
        label_ids += [self.tokenizer.eos_token_id]
        label_ids = label_ids[:self.args.max_dec_length] 
        label_ids += [-100] * (self.args.max_dec_length - len(label_ids))
        return label_ids

    def _get_raw_dataset(self, data_path):

        with open(data_path, 'r') as fr:
            lines = fr.readlines()

        if 'train' in data_path and self.args.train_ratio > 0:
            if self.train_sample_ids is None:
                self.train_sample_ids = set(random.sample(range(len(lines)), self.args.train_ratio))
            new_lines = [lines[line_id] for line_id in self.train_sample_ids]
        else:
            new_lines = lines

        line_iterator = tqdm(new_lines, desc='processing {}'.format(data_path))
        dataset = []
        for line in line_iterator:
            graph_obj = json.loads(line.strip())

            dataset.append(
                InputExample(
                    context=graph_obj['context'] if 'context' in graph_obj else None,
                    entities=graph_obj['entities'],
                    relations=graph_obj['relations'] if 'relations' in graph_obj else None,
                    text_label=self._get_tensor_text_label(graph_obj['text']) if 'text' in graph_obj else None,
                )
            )

        for example in dataset[:2]:
            print("*** Example ***")
            print(example)

        return TrainingDataset(dataset)

    def _get_tensor_dataset_for_inference(self, data_path):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []

        with open(data_path, 'r') as fr:
            line_iterator = tqdm(fr, desc='processing {}'.format(data_path))
            for line in line_iterator:
                graph_obj = json.loads(line.strip())
                context = graph_obj['context'] if 'context' in graph_obj else None
                input_seq = self.format_input(context, graph_obj['entities'], graph_obj['relations'])
                encoder_input = self.tokenizer(input_seq, padding='max_length', max_length=self.args.max_enc_length, truncation=True)
                encoder_input_ids = encoder_input['input_ids']
                encoder_attention_mask = encoder_input['attention_mask']

                encoder_input_tensor.append(encoder_input_ids)
                encoder_attention_mask_tensor.append(encoder_attention_mask)

        encoder_input_tensor = torch.tensor(encoder_input_tensor, dtype=torch.long)
        encoder_attention_mask_tensor= torch.tensor(encoder_attention_mask_tensor, dtype=torch.long)

        return TensorDataset(encoder_input_tensor, 
                            encoder_attention_mask_tensor)

    def format_input(self, context, entities, relations=None):
        input_seq = 'context: ' + context + ' ' if context is not None else ''
        input_seq += 'concepts: ' + '<SEP>'.join(entities)
        if self.args.method == 'ng2text':
            if relations is None:
                input_seq += ' relations: '
            else:
                input_seq += ' relations: ' + self._get_tensor_graph_input(relations)
        return input_seq

    def _get_tensor_dataset(self, data_path):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        text_label_tensor = []

        with open(data_path, 'r') as fr:
            lines = fr.readlines()

        if 'dev' in data_path and self.args.train_ratio > 0:
            lines = random.sample(lines, min(self.args.train_ratio, len(lines)))

        line_iterator = tqdm(lines, desc='processing {}'.format(data_path))
        for line in line_iterator:
            graph_obj = json.loads(line.strip())
            context = graph_obj['context'] if 'context' in graph_obj else None
            relations = graph_obj['relations'] if 'relations' in graph_obj else None
            input_seq = self.format_input(context, graph_obj['entities'], relations)
            encoder_input = self.tokenizer(input_seq, padding='max_length', max_length=self.args.max_enc_length, truncation=True)
            encoder_input_ids = encoder_input['input_ids']
            encoder_attention_mask = encoder_input['attention_mask']

            label_ids = self._get_tensor_text_label(graph_obj['text'])

            encoder_input_tensor.append(encoder_input_ids)
            encoder_attention_mask_tensor.append(encoder_attention_mask)
            text_label_tensor.append(label_ids)

        encoder_input_tensor = torch.tensor(encoder_input_tensor, dtype=torch.long)
        encoder_attention_mask_tensor= torch.tensor(encoder_attention_mask_tensor, dtype=torch.long)
        text_label_tensor = torch.tensor(text_label_tensor, dtype=torch.long)
        for f1, f2, f3  in zip(encoder_input_tensor[:2], 
                                    encoder_attention_mask_tensor[:2], 
                                    text_label_tensor[:2]):
            print("*** Example ***")
            print("encoder input: %s" % f1)
            print("encoder attention mask: %s" % f2)
            print("text label: %s" % f3)

        return TensorDataset(encoder_input_tensor, 
                            encoder_attention_mask_tensor,
                            text_label_tensor)

