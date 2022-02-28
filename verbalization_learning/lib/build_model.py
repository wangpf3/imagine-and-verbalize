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
from .utils.logging import get_logger
from .utils.setup_dist import cleanup

class Model:
    def __init__(self, args):
        self.args = args

        self.config = AutoConfig.from_pretrained(args.model_type, cache_dir='../cache/')
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_type, cache_dir='../cache')

        self.tokenizer.add_tokens(['<SEP>'])
        if not 'node' in self.args.method:
            with open('./data/{}/relation_vocab.json'.format(args.dataset), 'rb') as handle:
                relation_dict = json.load(handle)
            for reltype in relation_dict:
                self.tokenizer.add_tokens(['<{}>'.format(reltype)])
        tokenizer_size = max(len(self.tokenizer), self.config.vocab_size)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_type, cache_dir='../cache/')
        self.model.resize_token_embeddings(tokenizer_size)

    def train(self, seed):
        # ----------------------------------------------------- #
        # process raw data
        from .utils.data_helper import Data_Helper
        data_helper = Data_Helper(self.tokenizer, self.args, inference=False)
     
        # ----------------------------------------------------- #
        # prepare model and data loader
        if self.args.is_master:
            log_path = os.path.join(self.args.save_dir, 'train_seed{}.log'.format(seed))
            logger = get_logger("model", log_path)
            logger.info('args: {}'.format(self.args))
            logger = logging.getLogger("model.train")
            logger.info("Start training")

        model_ckpt = os.path.join(self.args.save_dir, 'model.seed{}.ckpt'.format(seed))

        self.model.to(self.args.device)

        if self.args.multi_gpu:
            self.model = DDP(self.model, device_ids=[self.args.local_rank])

        train_dataloaders = []
        train_sampler = RandomSampler(data_helper.trainset) if not self.args.multi_gpu else DistributedSampler(data_helper.trainset) 
        train_dataloaders.append(DataLoader(data_helper.trainset,
                    sampler=train_sampler, 
                    collate_fn=data_helper.data_collator,
                    batch_size=self.args.train_batch_size,
                    num_workers=self.args.num_workers
        ))

        if self.args.graph_source_alpha > 0:
            train_sampler = RandomSampler(data_helper.trainset_with_groundtruth) if not self.args.multi_gpu else DistributedSampler(data_helper.trainset_with_groundtruth) 
            train_dataloaders.append(DataLoader(data_helper.trainset_with_groundtruth,
                        sampler=train_sampler, 
                        collate_fn=data_helper.data_collator,
                        batch_size=self.args.train_batch_size,
                        num_workers=self.args.num_workers
            ))

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
        num_update_steps_per_epoch = min([len(train_dataloader) for train_dataloader in train_dataloaders])
        t_total = num_update_steps_per_epoch // self.args.grad_step * self.args.num_epoch
        warmup_steps = int(t_total * self.args.warmup_ratio)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps) #, num_training_steps=t_total)

        global_step = 0
        best_dev_loss = 1e19
        step_nogress = 0
        optimizer.zero_grad()
        if self.args.debug:
            self.args.num_epoch = 1
        for epoch in trange(int(self.args.num_epoch), desc="Epoch"):
            train_loss = 0.0
            self.model.train()
            if self.args.multi_gpu:
                for train_dataloader in train_dataloaders:
                    train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = tqdm(zip(*train_dataloaders), desc="Train Iteration at Epoch {}".format(epoch), disable=not self.args.is_master, total=num_update_steps_per_epoch)
            for step, batch_list in enumerate(epoch_iterator):

                batch_data = tuple(t.to(self.args.device) for t in batch_list[0]) 

                loss = 0.
                if self.args.graph_source_alpha < 1.0:
                    outputs = self.model(
                                input_ids=batch_data[0], 
                                attention_mask=batch_data[1], 
                                labels=batch_data[2],
                                output_hidden_states=True
                            )

                    loss += (1 - self.args.graph_source_alpha) * outputs.loss

                if self.args.graph_source_alpha > 0:
                    batch_data = tuple(t.to(self.args.device) for t in batch_list[1]) 

                    outputs_gt = self.model(
                                input_ids=batch_data[0], 
                                attention_mask=batch_data[1], 
                                labels=batch_data[2],
                                output_hidden_states=True
                            )

                    loss += self.args.graph_source_alpha * outputs_gt.loss

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
                if self.args.debug and global_step > 10:
                    break

            train_loss /= (step + 1)
            if self.args.is_master:
                log = 'Epoch: {:03d} Train loss: {:.4f}'
                logger.info(log.format(epoch, train_loss))

                dev_result = self.evaluate(data_helper, is_test=False)

                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                log = 'Epoch: {:03d}, dev loss {:.4f}, perplexity {:.4f}'
                if dev_result["loss"] <= best_dev_loss:
                    torch.save({'ckpt': model_to_save.state_dict(), 'args': self.args}, model_ckpt)
                    log += ' best'
                    best_dev_loss = dev_result["loss"]
                    step_nogress = 0
                else:
                    step_nogress += 1
                logger.info(log.format(epoch, dev_result["loss"], dev_result["perplexity"]))
                if step_nogress > 0:
                    break

            torch.cuda.empty_cache()

    def evaluate(self, data_helper, is_test=False):

        dataset = data_helper.testset if is_test else data_helper.devset
        data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, 
                        sampler=data_sampler, 
                        collate_fn=None,
                        batch_size=self.args.eval_batch_size)
        self.model.eval()
        epoch_iterator = tqdm(dataloader, desc="Eval Iteration")

        loss_sum = 0.
        ppl_sum = 0.
        tokens_sum = 0.
        for step, batch in enumerate(epoch_iterator):

            input_ids, attention_mask, text_labels = tuple(t.to(self.args.device) for t in batch) 

            with torch.no_grad():
                outputs = self.model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=text_labels
                        )

                loss = outputs.loss
                num_tokens = (text_labels != -100).sum().item()
                tokens_sum += num_tokens
                ppl_sum += outputs.loss.item() * num_tokens

                loss_sum += loss.item()
            if self.args.debug and step > 100:
                break

        loss_sum /= (step + 1)
        ppl_sum = math.exp(ppl_sum / tokens_sum)

        return {"loss": loss_sum, "perplexity": ppl_sum}

    def generate(self, output_path, args, split='test'):
        print('device:', args.device)
        self.model.to(args.device)
        self.model.eval()

        from .utils.data_helper import Data_Helper
        data_helper = Data_Helper(self.tokenizer, args, split, inference=True)
        dataset = data_helper.testset
        data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.eval_batch_size, 
                                collate_fn=data_helper.data_collator_for_concept2sentence_inference)
        epoch_iterator = tqdm(dataloader, desc="Generate Iteration")

        output_file = open(output_path, 'w')
        for step, batch in enumerate(epoch_iterator):

            batch = tuple(t.to(args.device) for t in batch) 
            input_ids = batch[0]
            attention_mask = batch[1]
            with torch.no_grad():

                text_pred_ids = self.model.generate(
                      input_ids=input_ids, 
                      attention_mask=attention_mask,
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
                gen_texts = self.tokenizer.batch_decode(text_pred_ids, skip_special_tokens=True)

            for text in gen_texts:
                example = {'text': text}
                output_file.write(json.dumps(example) + '\n')

            if self.args.debug and step > 10:
                break

        output_file.close()

    def generate_story_with_dynamic_graph(self, output_path, args, split='test', graph_generator=None, num_target_sent=4):
        print('device:', args.device)
        self.model.to(args.device)
        self.model.eval()
        graph_device = torch.device('cuda:1') if args.multi_gpu else torch.device('cuda:0')
        graph_generator.model.to(graph_device)
        graph_generator.model.eval()

        from .utils.data_helper import Data_Helper
        data_helper = Data_Helper(self.tokenizer, args, split, inference=True)
        dataset = data_helper.testset
        num_batch = int(len(dataset) / args.eval_batch_size)
        epoch_iterator = tqdm(data_helper.sequential_iterate(dataset, args.eval_batch_size), desc="Generate Iteration", total=num_batch)

        output_file = open(output_path, 'w')
        for step, batch in enumerate(epoch_iterator):
            assert len(batch) % num_target_sent == 0
            batch_story = []
            batch_story_graph = []
            for sent_id in range(num_target_sent):
                batch_example = batch[sent_id::num_target_sent]
                if sent_id == 0:
                    generated_context = [feature.context for feature in batch_example]
                input_ids, attention_mask = data_helper.data_collator_for_concept2graph_inference(graph_generator.tokenizer, batch_example, generated_context)

                with torch.no_grad():

                    graph_pred_ids = graph_generator.model.generate(
                          input_ids=input_ids.to(graph_device), 
                          attention_mask=attention_mask.to(graph_device),
                          max_length=graph_generator.args.max_dec_length,
                          eos_token_id=graph_generator.tokenizer.eos_token_id, 
                          pad_token_id=graph_generator.tokenizer.pad_token_id,
                          early_stopping=True,
                          use_cache=True,
                          do_sample=args.sample,
                          num_return_sequences=args.num_return_sequences,
                          num_beams=args.num_beams,
                          top_p=args.top_p,
                          top_k=args.top_k,
                         ) 
                generated_graph = graph_generator.tokenizer.batch_decode(graph_pred_ids, skip_special_tokens=True)
                batch_story_graph.append(generated_graph)

                input_ids, attention_mask = data_helper.data_collator_for_graph2story_inference(batch_example, generated_context, generated_graph)
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
                gen_texts = self.tokenizer.batch_decode(text_pred_ids, skip_special_tokens=True)
                batch_story.append(gen_texts)
                generated_context = [context + ' ' + generated for context, generated in zip(generated_context, gen_texts)]
                # torch.cuda.empty_cache()

            batch_story = np.asarray(batch_story)
            batch_story_graph = np.asarray(batch_story_graph)
            for story_id in range(len(batch_story[0])):
                sents = batch_story[:, story_id]
                graphs = batch_story_graph[:, story_id]
                for text, graph in zip(sents, graphs):
                    example = {'text': text, 'relations': graph}
                    output_file.write(json.dumps(example) + '\n')

            if self.args.debug and step > 1:
                break

        output_file.close()
