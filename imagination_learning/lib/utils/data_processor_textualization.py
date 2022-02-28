import json
import os 
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset, TensorDataset

@dataclass(frozen=True)
class InputExample:

    context: str
    entities: List[str]
    label: List[int]

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

    def load_trainset(self,):
        return self._get_raw_dataset(os.path.join(self.data_dir, "train.json"))

    def load_devset(self,):
        return self._get_tensor_dataset(os.path.join(self.data_dir, "dev.json"))

    def load_testset(self,):
        return self._get_tensor_dataset(os.path.join(self.data_dir, "test.json"))

    def load_datasplit(self, split):
        return self._get_raw_dataset(os.path.join(self.data_dir, "{}.json".format(split)))

    def _get_tensor_label(self, raw_label):
        label = ''
        if len(raw_label) != 0:
            triplet = raw_label[0]
            label += triplet[0] + '<{}>'.format(triplet[1].replace('<', '').replace('>', '')) + triplet[2]

            for triplet in raw_label[1:]:
                label += '<SEP>' + triplet[0] + '<{}>'.format(triplet[1]) + triplet[2]

        label_ids = self.tokenizer.encode(label, add_special_tokens=False)
        label_ids += [self.tokenizer.eos_token_id]
        label_ids = label_ids[:self.args.max_dec_length]
        label_ids += [-100] * (self.args.max_dec_length - len(label_ids))
        return label_ids

    def _get_raw_dataset(self, data_path):
        dataset = []

        with open(data_path, 'r') as fr:
            line_iterator = tqdm(fr, desc='processing {}'.format(data_path))
            for line in line_iterator:
                graph_obj = json.loads(line.strip())
                if 'train' in data_path:
                    if len(graph_obj['relations']) == 0:
                        continue

                dataset.append(
                    InputExample(
                        context=graph_obj['context'] if 'context' in graph_obj else None,
                        entities=graph_obj['entities'],
                        label=self._get_tensor_label(graph_obj['relations']) if 'relations' in graph_obj else None
                    )
                )

        for example in dataset[:2]:
            print("*** Example ***")
            print(example)

        return TrainingDataset(dataset)

    def _get_tensor_text_label(self, raw_label):
        label_ids = self.tokenizer.encode(raw_label, add_special_tokens=False)
        label_ids += [self.tokenizer.eos_token_id]
        label_ids = label_ids[:self.args.max_dec_length] 
        label_ids += [-100] * (self.args.max_dec_length - len(label_ids))
        return label_ids

    def _get_context_tensor(self, context, entities):

        if context == "" or context is None:
            context = 'none'

        input_seq = 'context: ' + context + ' entities: ' + '<SEP>'.join(entities)
        input_ids = self.tokenizer.encode(input_seq, add_special_tokens=True)
        input_ids = input_ids[-self.args.max_enc_length:]
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_enc_length - len(input_ids))
        attention_mask = [0 if _id == self.tokenizer.pad_token_id else 1 for _id in input_ids]
        return input_ids, attention_mask

    def _get_tensor_dataset(self, data_path):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        label_tensor = []

        with open(data_path, 'r') as fr:
            line_iterator = tqdm(fr, desc='processing {}'.format(data_path))
            for line in line_iterator:
                graph_obj = json.loads(line.strip())
                context = graph_obj['context'] if 'context' in graph_obj else None
                encoder_input_ids, encoder_attention_mask = self._get_context_tensor(context, graph_obj['entities'])
                label_ids = self._get_tensor_label(graph_obj['relations'])

                encoder_input_tensor.append(encoder_input_ids)
                encoder_attention_mask_tensor.append(encoder_attention_mask)
                label_tensor.append(label_ids)

        encoder_input_tensor = torch.tensor(encoder_input_tensor, dtype=torch.long)
        encoder_attention_mask_tensor= torch.tensor(encoder_attention_mask_tensor, dtype=torch.long)
        label_tensor = torch.tensor(label_tensor, dtype=torch.long)
        for f1, f2, f3 in zip(encoder_input_tensor[:2], encoder_attention_mask_tensor[:2], label_tensor[:2]):
            print("*** Example ***")
            print("encoder input: %s" % f1)
            print("encoder attention mask: %s" % f2)
            print("label: %s" % f3)

        return TensorDataset(encoder_input_tensor, encoder_attention_mask_tensor, label_tensor)

    def _get_tensor_dataset_for_inference(self, data_path):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []

        with open(data_path, 'r') as fr:
            line_iterator = tqdm(fr, desc='processing {}'.format(data_path))
            for line in line_iterator:
                graph_obj = json.loads(line.strip())
                context = graph_obj['context'] if 'context' in graph_obj else None
                encoder_input = self.tokenizer('context: ' + context +  ' entities: ' + '<SEP>'.join(graph_obj['entities']), padding='max_length', max_length=self.args.max_enc_length, truncation=True)
                encoder_input_ids = encoder_input['input_ids']
                encoder_attention_mask = encoder_input['attention_mask']

                encoder_input_tensor.append(encoder_input_ids)
                encoder_attention_mask_tensor.append(encoder_attention_mask)

        encoder_input_tensor = torch.tensor(encoder_input_tensor, dtype=torch.long)
        encoder_attention_mask_tensor= torch.tensor(encoder_attention_mask_tensor, dtype=torch.long)

        return TensorDataset(encoder_input_tensor, 
                            encoder_attention_mask_tensor)
