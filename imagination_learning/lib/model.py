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

from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from .utils.data_helper import Data_Helper
from .utils.logging import get_logger
from .utils.loss import LabelSmoothingLoss

class Build_Model:
    def __init__(self, args):
        self.args = args

        self.config = AutoConfig.from_pretrained(args.model_type, cache_dir='../cache/')
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_type, cache_dir='../cache')

        self.tokenizer.add_tokens(['<SEP>'])
        if args.pretrain_dir != 'none':
            with open('{}/relation_vocab.json'.format(args.pretrain_dir), 'rb') as handle:
                relation_dict = json.load(handle)
                print('num of relations:', len(relation_dict))
        else:
            with open('./data/{}/relation_vocab.json'.format(args.dataset), 'rb') as handle:
                relation_dict = json.load(handle)
        for reltype in relation_dict:
            self.tokenizer.add_tokens(['<{}>'.format(reltype)])


        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_type, cache_dir='../cache/')
        self.model.resize_token_embeddings(len(self.tokenizer))

        if hasattr(self.args, 'pretrain_dir') and self.args.pretrain_dir != 'none' and args.do_train:
            ckpt_path = os.path.join(self.args.pretrain_dir, 'model.ckpt.best')
            print('initialzing model with pretrain ckpt from', ckpt_path)
            pretrain_ckpt = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(pretrain_ckpt['ckpt'])

    def train(self):

        if self.args.pretrain_dir != 'none':

            self.args.save_dir = os.path.join(self.args.save_dir, 'from_pretrain')
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
 
        if self.args.is_master:
            log_path = os.path.join(self.args.save_dir, 'train.log')
            logger = get_logger("model", log_path)
            logger.info('args: {}'.format(self.args))
            logger = logging.getLogger("model.train")
            logger.info("Start training")

        # ----------------------------------------------------- #

        if self.args.multi_gpu and not self.args.is_master:
            torch.distributed.barrier() 

        data_helper = Data_Helper(self.tokenizer, self.args)

        if self.args.multi_gpu and self.args.is_master:
            torch.distributed.barrier() 

        model_ckpt = os.path.join(self.args.save_dir, 'model.ckpt')

        if self.args.multi_gpu:
            train_sampler = DistributedSampler(data_helper.trainset)
        else:
            train_sampler = RandomSampler(data_helper.trainset)
        train_dataloader = DataLoader(data_helper.trainset,
                    sampler=train_sampler, 
                    collate_fn=data_helper.data_collator,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
        )

        self.model.to(self.args.device)
        if self.args.multi_gpu:
            self.model = DDP(self.model, device_ids=[self.args.local_rank])

        # ----------------------------------------------------- #
        # setup optimization

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        t_total = len(train_dataloader) // self.args.grad_step * self.args.num_epoch
        warmup_steps = t_total * self.args.warmup_ratio
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps) #, num_training_steps=t_total)

        global_step = 0
        best_recall = 0.
        best_dev_perplexity = 1e19
        step_nogress = 0
        optimizer.zero_grad()
        loss_fn = LabelSmoothingLoss(smoothing=self.args.smooth_factor)
        if self.args.debug:
            self.args.num_epoch = 2
        for epoch in trange(int(self.args.num_epoch), desc="Epoch"):
            train_loss = 0.0
            self.model.train()
            epoch_iterator = tqdm(train_dataloader, desc="Train Iteration at Epoch {}".format(epoch), disable=not self.args.is_master)
            for step, batch in enumerate(epoch_iterator):

                input_ids, attention_mask, labels = tuple(t.to(self.args.device) for t in batch) 
                outputs = self.model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=labels
                        )
                nll_loss = outputs.loss
                lm_logits = outputs.logits
                loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), nll_loss)

                loss /= self.args.grad_step
                loss.backward()
                if (global_step + 1) % self.args.grad_step == 0:

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()

                train_loss += loss.item() * self.args.grad_step
                global_step += 1
                epoch_iterator.set_description("Epoch {} loss {:.4f}".format(epoch, train_loss / (step + 1)))
                if self.args.debug and global_step > 50:
                    break

            train_loss /= (step + 1)
            if self.args.is_master:
                log = 'Epoch: {:03d} Train loss: {:.4f}'
                logger.info(log.format(epoch, train_loss))
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                torch.save({'ckpt': model_to_save.state_dict(), 'args': self.args}, model_ckpt + "{}".format(epoch))

                dev_result = self.evaluate(data_helper, is_test=False)

                log = 'Epoch: {:03d}, dev loss {:.4f}, perplexity {:.4f}'
                if dev_result["perplexity"] <= best_dev_perplexity:
                    torch.save({'ckpt': model_to_save.state_dict(), 'args': self.args}, model_ckpt + ".best")
                    log += ' best'
                    best_dev_perplexity = dev_result["perplexity"]
                    step_nogress = 0
                else:
                    step_nogress += 1
                logger.info(log.format(epoch, dev_result["loss"], dev_result["perplexity"]))
                if step_nogress > 1:
                    break
            torch.cuda.empty_cache()

    def evaluate(self, data_helper, is_test=False):

        dataset = data_helper.testset if is_test else data_helper.devset
        data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=self.args.batch_size)
        self.model.eval()
        epoch_iterator = tqdm(dataloader, desc="Eval Iteration")

        loss_sum = 0.
        ppl_sum = 0.
        tokens_sum = 0.
        for step, batch in enumerate(epoch_iterator):

            input_ids, attention_mask, labels = tuple(t.to(self.args.device) for t in batch) 
            with torch.no_grad():
                outputs = self.model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=labels
                        )
                loss = outputs.loss
                num_tokens = (labels != -100).sum().item()
                tokens_sum += num_tokens
                ppl_sum += loss.item() * num_tokens

                loss_sum += loss.item()
            if self.args.debug and step > 50:
                break

        loss_sum /= (step + 1)
        ppl_sum = math.exp(ppl_sum / tokens_sum)

        return {"loss": loss_sum, "perplexity": ppl_sum}

    def generate(self, output_path, args, split='test'):
        self.model.to(args.device)
        self.model.eval()

        data_helper = Data_Helper(self.tokenizer, args, split)
        dataset = data_helper.testset
        data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.eval_batch_size,
                            collate_fn=data_helper.data_collator_for_inference)
        epoch_iterator = tqdm(dataloader, desc="Generate Iteration")

        output_file = open(output_path, 'w')
        for step, batch in enumerate(epoch_iterator):

            input_ids, attention_mask = tuple(t.to(args.device) for t in batch) 
            with torch.no_grad():
                pred_ids = self.model.generate(
                      input_ids=input_ids, 
                      attention_mask=attention_mask,
                      max_length=self.args.max_dec_length,
                      decoder_start_token_id=self.model.config.decoder_start_token_id,
                      eos_token_id=self.tokenizer.eos_token_id, 
                      pad_token_id=self.tokenizer.pad_token_id,
                      early_stopping=True, num_return_sequences=args.num_return_sequences,
                      num_beams=args.num_beams,
                      do_sample=args.sample,
                      top_p=args.top_p,
                      top_k=args.top_k,
                      use_cache=True
                     ) 

            for beam in pred_ids:
                gen = self.tokenizer.decode(beam, skip_special_tokens=True)
                # output_file.write(','.join(gen.split(self.tokenizer.sep_token)) + '\n')
                output_file.write(gen + '\n')

        output_file.close()

    def generate_contextualized(self, output_path, args, split='test', num_target_sent=4):
        self.model.to(args.device)
        self.model.eval()

        from .utils.data_helper import Data_Helper
        data_helper = Data_Helper(self.tokenizer, args, split)
        dataset = data_helper.testset
        num_batch = int(len(dataset) / args.eval_batch_size)
        epoch_iterator = tqdm(data_helper.sequential_iterate(dataset, args.eval_batch_size), desc="Generate Iteration", total=num_batch)

        output_file = open(output_path, 'w')
        for step, batch in enumerate(epoch_iterator):
            assert len(batch) % num_target_sent == 0
            batch_story_graph = []
            for sent_id in range(num_target_sent):
                batch_example = batch[sent_id::num_target_sent]
                if sent_id < 1 or args.textualization or args.is_training:
                    generated_context = [feature.context for feature in batch_example]
                input_ids, attention_mask = data_helper.data_collator_for_inference_contextualized(batch_example, generated_context)
                with torch.no_grad():

                    text_pred_ids = self.model.generate(
                          input_ids=input_ids.to(args.device), 
                          attention_mask=attention_mask.to(args.device),
                          max_length=args.max_dec_length,
                          eos_token_id=self.tokenizer.eos_token_id, 
                          pad_token_id=self.tokenizer.pad_token_id,
                          early_stopping=True,
                          use_cache=True,
                          do_sample=args.sample,
                          num_return_sequences=args.num_return_sequences,
                          num_beams=args.num_beams,
                          top_p=args.top_p,
                          top_k=args.top_k,
                         ) 

                generated_context = self.tokenizer.batch_decode(text_pred_ids, skip_special_tokens=True)
                batch_story_graph.append(generated_context)

            batch_story_graph = np.asarray(batch_story_graph)
            for story_id in range(len(batch_story_graph[0])):
                graphs = batch_story_graph[:, story_id]
                for g in graphs:
                    output_file.write(g + '\n')

            if self.args.debug and step > 1:
                break

        output_file.close()
