import os
import logging
from tqdm import tqdm, trange
import numpy as np
import json

import torch
import torch.nn.functional as F

def parse_graph(graph):
    triplet_prediction = []
    cur_subject, cur_relation, cur_object = None, None, None
    for element in graph.split('<SEP>'):
        sub_element = element.split('<')
        if len(sub_element) < 2:
            continue
        cur_subject = sub_element[0].strip()
        sub_element = sub_element[1].split('>')
        if len(sub_element) < 2:
            continue
        cur_relation = sub_element[0].strip()
        cur_object = sub_element[1].strip()
        triplet_prediction.append([cur_subject, cur_relation, cur_object])
        cur_subject, cur_relation, cur_object = None, None, None
    return triplet_prediction

def parse_and_save(graph_path, text_path, output_path):
    output_file = open(output_path, 'w')
    with open(graph_path, 'r') as graph_file, open(text_path, 'r') as text_file:
        for graph_line, text_line in zip(graph_file, text_file):
            text_example = json.loads(text_line.strip())
            graph = graph_line.strip()
            text_example['relations'] = parse_graph(graph)
            # text_example.pop('context', None)
            output_file.write(json.dumps(text_example)+'\n')
    output_file.close()
