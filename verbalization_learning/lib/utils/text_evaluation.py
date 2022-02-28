from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

import json
import spacy
nlp = spacy.load("en_core_web_sm")

def tokenize(dict):
    for key in dict:
        new_sentence_list = []
        for sentence in dict[key]:
            a = ''
            for token in nlp(sentence):
                a += token.text
                a += ' '
            new_sentence_list.append(a.rstrip())
        dict[key] = new_sentence_list

    return dict

def evaluator(gts, res):
    eval = {}
    # =================================================
    # Set up scorers
    # =================================================
    # Todo: use Spacy for tokenization
    gts = tokenize(gts)
    res = tokenize(res)

    # =================================================
    # Set up scorers
    # =================================================
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    # =================================================
    # Compute scores
    # =================================================
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                eval[m] = sc
        else:
            eval[method] = score
    return eval

def evaluate_sentence(gen_path, ref_path):
    gts = {}
    res = {}
    with open(ref_path, 'r') as f:
        gts_lines = f.readlines()
    with open(gen_path, 'r') as f:
        res_lines = f.readlines()

    for gts_line, res_line in zip(gts_lines, res_lines):
        sample = json.loads(gts_line.strip())
        generation = json.loads(res_line.strip())
        key = '#'.join(sorted(sample['entities']))
        if key not in gts:
            gts[key] = []
            gts[key].append(sample['text'])
            res[key] = []
            res[key].append(generation['text'])
        else:
            gts[key].append(sample['text'])
    return evaluator(gts, res)

def evaluate_story(gen_path, ref_path):
    gts = {}
    res = {}
    with open(ref_path, 'r') as f:
        gts_lines = f.readlines()
    with open(gen_path, 'r') as f:
        res_lines = f.readlines()

    for gts_line, res_line in zip(gts_lines, res_lines):
        sample = json.loads(gts_line.strip())
        generation = json.loads(res_line.strip())
        key = sample['id']
        # if key not in gts:
        gts[key] = []
        # gts[key].append(' '.join(sample['text_by_sent'][1:]))
        gts[key].append(sample['text'])
        res[key] = []
        res[key].append(generation['text'])
        # else:
        #     gts[key].append(sample['text'])
    return evaluator(gts, res)

if __name__ == "__main__":
    gts = {"cat#dog#boy": ["The dog is the boy's cat.", "The dog eats the cat of the boy."],
           "apple#tree#boy": ["A boy is picking apples from trees."]}
    res = {"cat#dog#boy": ["The dog is the boy's cat."],
           "apple#tree#boy": ["A boy is picking apples from trees and put them into bags."]}
    print(evaluator(gts, res))