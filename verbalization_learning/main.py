import os
import argparse
import numpy as np
import json
import torch
import random
from collections import defaultdict

from lib.utils.setup_dist import setup, cleanup

# torch.set_num_threads(4)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def train(args, seed):

    from lib.build_model import Model 

    model = Model(args)
    model.train(seed)
    return model

def inference(args, seed, model=None):
    model_ckpt = os.path.join(args.save_dir, 'model.seed{}.ckpt'.format(seed))
    assert os.path.exists(model_ckpt)
    save_ckpt = torch.load(model_ckpt, map_location='cpu')
    old_args = save_ckpt['args']
    args.max_enc_length = old_args.max_enc_length
    args.max_dec_length = old_args.max_dec_length
    args.method = old_args.method

    old_args.device = args.device

    evaluation_result = {}
    for split in args.eval_split.split(','):
        save_prefix = '_{}_{}_beam{}_seed{}-{}-{}'.format(args.dataset, split, args.num_beams, args.data_seed, seed, args.ckpt_num)
        output_path = os.path.join(args.save_dir, 'generation{}.txt'.format(save_prefix))
        if not os.path.exists(output_path) or args.overwrite_output:
            if model is None:
                from lib.build_model import Model 
                model = Model(old_args)
            if hasattr(model.model, 'module'):
                model.module.load_state_dict(save_ckpt['ckpt'])
            else:
                model.model.load_state_dict(save_ckpt['ckpt'])
            if any(_n in args.dataset for _n in ['vist', 'roc']):
                from lib.build_graph_generator import Build_GraphGenerator
                graph_generator = Build_GraphGenerator(args)
                model.generate_story_with_dynamic_graph(output_path, args, split, graph_generator)
            else:
                model.generate(output_path, args, split)
        if args.evaluate:
            reference_path = os.path.join('./data', args.dataset, '{}.json'.format(split))
            try:
                if any(_n in args.dataset for _n in ['vist', 'roc']):
                    from lib.utils.text_evaluation import evaluate_story as evaluate_text
                else:
                    from lib.utils.text_evaluation import evaluate_sentence as evaluate_text
                split_result = evaluate_text(output_path, reference_path)
                evaluation_result[split] = split_result
                with open(os.path.join(args.save_dir, 'evaluation{}.json'.format(save_prefix)), 'w') as fw:
                    json.dump(split_result, fw, indent=4)
            except Exception as e:
                print(e)
                print('Not evaluated')

    print('Generation / Evaluation on seed {} finished!'.format(seed))
    return evaluation_result

def summarize_result(overall_result, args):
    for split in overall_result[0]:
        score_list_by_metric = defaultdict(list)
        for seed_result in overall_result:
            for metric in seed_result[split]:
                score_list_by_metric[metric].append(seed_result[split][metric])
        save_result = {}
        for metric in score_list_by_metric:
            mean_score = np.mean(score_list_by_metric[metric])
            std_score = np.std(score_list_by_metric[metric])
            save_result[metric] = {'mean': mean_score, 'std': std_score}
        result_path = os.path.join(args.save_dir, 'evaluation_{}_{}_beam{}_average{}'.format(args.dataset, split, args.num_beams, args.episodes))
        if args.graph_generator_dir is not None:
            result_path += '_textualization'
        result_path += '.json'
        with open(result_path, 'w') as fw:
                json.dump(save_result, fw, indent=4)

def parse_seeds(seed_str):

    seed_list = [int(ep) for ep in seed_str.split('-')]
    if len(seed_list) == 1:
        random_seeds = seed_list
    else:
        start_seed, end_seed = seed_list
        random_seeds = list(range(start_seed, end_seed + 1))
    return random_seeds

def main():

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--pretrain_dir", type=str, default=None)
    parser.add_argument('--project_dir', type=str)
    parser.add_argument('--graph_generator_dir', type=str, default=None)
    parser.add_argument('--episodes', type=str)
    parser.add_argument('--data_seed', type=int)
    parser.add_argument('--train_ratio', default=0, type=int)
    parser.add_argument("--do_train", action='store_true')

    parser.add_argument('--model_type', type=str)
    parser.add_argument('--max_enc_length', type=int, default=128)
    parser.add_argument('--max_dec_length', type=int, default=64)

    # concept2story format
    parser.add_argument('--method', type=str)

    # mix groundtruth and generated graph
    parser.add_argument('--graph_source_alpha', default=0, type=float)

    # optimization
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--grad_step', type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument('--num_epoch', type=float, default=1000)

    # inference
    parser.add_argument('--ckpt_num', type=int, default=-1)
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument("--overwrite_output", action='store_true')

    # gpu and workers option
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--master_port', type=int, default=-1)
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    setup(args)

    if args.local_rank == -1:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cuda:{}'.format(args.local_rank))


    train_seeds = parse_seeds(args.episodes)

    overall_result = []
    # for REPRODUCIBILITY
    for seed_j in train_seeds:
        model = None
        np.random.seed(args.data_seed)
        random.seed(args.data_seed)
        torch.manual_seed(seed_j)
        torch.cuda.manual_seed(seed_j)
        if args.do_train:
            model = train(args, seed_j)

        if args.local_rank in [-1, 0] and args.inference:
            overall_result.append(inference(args, seed_j, model))

        # cleanup()
        # torch.cuda.empty_cache()

    if args.local_rank in [-1, 0] and len(overall_result) > 1:
        summarize_result(overall_result, args)

if __name__ == "__main__":
    main()

