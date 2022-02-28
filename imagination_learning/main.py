import os
import argparse
import numpy as np
import json
import torch

from lib.utils.logging import get_logger
from lib.utils.graph_evaluation import evaluate_graph, parse_and_save
from lib.model import Build_Model 
from lib.utils.setup_dist import setup, cleanup

# torch.set_num_threads(4)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# for REPRODUCIBILITY
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(args):
    model = Build_Model(args)
    model.train()

def inference(args, model=None):
    if args.ckpt_num == -1:
        model_ckpt = os.path.join(args.save_dir, 'model.ckpt.best')
    else:
        model_ckpt = os.path.join(args.save_dir, 'model.ckpt{}'.format(args.ckpt_num))
    assert os.path.exists(model_ckpt)
    print("loading model from", model_ckpt)
    save_ckpt = torch.load(model_ckpt, map_location='cpu')
    old_args = save_ckpt['args']
    args.max_enc_length = old_args.max_enc_length
    args.max_dec_length = old_args.max_dec_length
    args.separation = old_args.separation
    old_args.inference = True

    for split in args.eval_split.split(','):
        output_path = os.path.join(args.save_dir, 'generation_{}_{}_beam{}_ckpt{}.txt'.format(args.dataset, split, args.num_beams, args.ckpt_num))
        if not os.path.exists(output_path) or args.overwrite_output:
            if model is None:
                model = Build_Model(old_args)
            model.model.load_state_dict(save_ckpt['ckpt'])
            if any(_n in args.dataset for _n in ['vist', 'roc']):
                model.generate_contextualized(output_path, args, split)
            else:
                model.generate(output_path, args, split)
        print('Generation finished: {}'.format(split))

        if args.parse_output:
            reference_path = os.path.join('./data', args.dataset, '{}.json'.format(split))
            save_path = os.path.join(args.save_dir, '{}.{}.{}.beam{}_ckpt{}.json'.format(args.dataset, split, old_args.model_type.replace('/', '-'), args.num_beams, args.ckpt_num))
            parse_and_save(output_path, reference_path, save_path)

def main():

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--pretrain_dir', type=str)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--textualization', action='store_true')
    parser.add_argument('--is_training', action='store_true')

    parser.add_argument('--model_type', type=str)
    parser.add_argument('--max_enc_length', type=int, default=128)
    parser.add_argument('--max_dec_length', type=int, default=128)
    parser.add_argument('--separation', type=str, default='sep')
    parser.add_argument('--mask_ratio', type=float, default=0)

    # optimization
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--grad_step', type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument('--num_epoch', type=float, default=1000)
    parser.add_argument('--smooth_factor', type=float, default=0)

    # inference
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--parse_output", action='store_true')
    parser.add_argument("--overwrite_output", action='store_true')
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--ckpt_num', type=int)

    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)

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
 
    model = None
    if args.do_train:
        train(args)
    if args.is_master and args.inference:
        inference(args, model)

if __name__ == "__main__":
    main()

