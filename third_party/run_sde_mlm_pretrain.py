# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pretrain SDE model."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils_tag import create_sde_pretrain_vocab_features 
from utils_tag import get_labels
from utils_tag import read_examples_from_file

from transformers import (
  AdamW,
  get_linear_schedule_with_warmup,
  WEIGHTS_NAME,
  BertConfig,
  BertTokenizer,
  SdeBertTokenizer,
  BertForTokenClassification,
  BertForMaskedLM,
  XLMConfig,
  XLMTokenizer,
  XLMRobertaConfig,
  XLMRobertaTokenizer,
  XLMRobertaForTokenClassification
)
from xlm import XLMForTokenClassification


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
  (tuple(conf.pretrained_config_archive_map.keys())
    for conf in (BertConfig, XLMConfig, XLMRobertaConfig)),
  ()
)

MODEL_CLASSES = {
  "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
  "xlm": (XLMConfig, XLMForTokenClassification, XLMTokenizer),
  "xlmr": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, sde_model, model, tokenizer, labels, pad_token_label_id, lang2id=None):
  """Train the model."""
  if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter()

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
  else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {"params": [p for n, p in sde_model.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": args.weight_decay},
    {"params": [p for n, p in sde_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    sde_model, optimizer = amp.initialize(sde_model, optimizer, opt_level=args.fp16_opt_level)

  # multi-gpu training (should be after apex fp16 initialization)
  if args.n_gpu > 1:
    sde_model = torch.nn.DataParallel(sde_model)

  # Distributed training (should be after apex fp16 initialization)
  if args.local_rank != -1:
    sde_model = torch.nn.parallel.DistributedDataParallel(sde_model, device_ids=[args.local_rank],
                              output_device=args.local_rank,
                              find_unused_parameters=True)

  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps * (
          torch.distributed.get_world_size() if args.local_rank != -1 else 1))
  logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  logger.info("  Total optimization steps = %d", t_total)

  best_score = 10000
  best_checkpoint = None
  patience = 0
  global_step = 0
  tr_loss, logging_loss = 0.0, 0.0
  sde_model.zero_grad()
  train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
  set_seed(args) # Add here for reproductibility (even between python 2 and 3)

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
      sde_model.train()
      batch = tuple(t.to(args.device) for t in batch if t is not None)
      input_ids = batch[0]
      labels = batch[3]
      sde_embedding = sde_model.bert.embeddings.sde_word_embeddings(input_ids) 
      subword_embedding = model.bert.embeddings.word_embeddings(labels)
      mse_loss = torch.nn.MSELoss(reduction='none')
      loss = mse_loss(sde_embedding, subword_embedding)
      num_words = labels.numel()
      loss = loss.sum() / num_words

      if args.n_gpu > 1:
        # mean() to average on multi-gpu parallel training
        loss = loss.mean()
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

      if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()

      tr_loss += loss.item()
      if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        scheduler.step()  # Update learning rate schedule
        optimizer.step()
        sde_model.zero_grad()
        global_step += 1

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          # Log metrics
          if args.local_rank == -1 and args.evaluate_during_training:
            # Only evaluate on single GPU otherwise metrics may not average well
            results, _ = evaluate(args, sde_model, model, tokenizer, labels, pad_token_label_id, mode="dev", lang=args.train_langs, lang2id=lang2id)
            for key, value in results.items():
              tb_writer.add_scalar("eval_{}".format(key), value, global_step)
          tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
          tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
          logging_loss = tr_loss

        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
          if args.save_only_best_checkpoint:
            result, _ = evaluate(args, sde_model, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step, lang=args.train_langs, lang2id=lang2id)
            if result["loss"] < best_score:
              logger.info("result['loss']={} < best_score={}".format(result["loss"], best_score))
              best_score = result["loss"]
              # Save the best model checkpoint
              output_dir = os.path.join(args.output_dir, "checkpoint-best")
              best_checkpoint = output_dir
              if not os.path.exists(output_dir):
                os.makedirs(output_dir)
              # Take care of distributed/parallel training
              model_to_save = sde_model.module if hasattr(sde_model, "module") else sde_model
              model_to_save.save_pretrained(output_dir)
              torch.save(args, os.path.join(output_dir, "training_args.bin"))
              logger.info("Saving the best model checkpoint to %s", output_dir)
              logger.info("Reset patience to 0")
              patience = 0
            else:
              patience += 1
              logger.info("Hit patience={}".format(patience))
              if args.eval_patience > 0 and patience > args.eval_patience:
                logger.info("early stop! patience={}".format(patience))
                epoch_iterator.close()
                train_iterator.close()
                if args.local_rank in [-1, 0]:
                  tb_writer.close()
                return global_step, tr_loss / global_step
          else:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
              os.makedirs(output_dir)
            # Take care of distributed/parallel training
            model_to_save = sde_model.module if hasattr(sde_model, "module") else sde_model
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

      if args.max_steps > 0 and global_step > args.max_steps:
        epoch_iterator.close()
        break
    if args.max_steps > 0 and global_step > args.max_steps:
      train_iterator.close()
      break

  if args.local_rank in [-1, 0]:
    tb_writer.close()

  return global_step, tr_loss / global_step


def evaluate(args, sde_model, model, tokenizer, labels, pad_token_label_id, mode, prefix="", lang="en", lang2id=None, print_result=True):
  eval_dataset = load_and_cache_examples(args, tokenizer)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
  # Note that DistributedSampler samples randomly
  eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
  eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

  # multi-gpu evaluate
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  # Eval!
  logger.info("***** Running evaluation %s in %s *****" % (prefix, lang))
  logger.info("  Num examples = %d", len(eval_dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)
  eval_loss = 0.0
  nb_eval_steps = 0
  preds = None
  out_label_ids = None
  model.eval()
  sde_model.eval()
  for batch in tqdm(eval_dataloader, desc="Evaluating"):
    batch = tuple(t.to(args.device) for t in batch)

    with torch.no_grad():
      input_ids = batch[0]
      labels = batch[3]
      sde_embedding = sde_model.bert.embeddings.sde_word_embeddings(input_ids) 
      subword_embedding = model.bert.embeddings.word_embeddings(labels)
      mse_loss = torch.nn.MSELoss(reduction='none')
      loss = mse_loss(sde_embedding, subword_embedding)
      num_words = labels.numel()
      tmp_eval_loss = loss.sum() / num_words

      eval_loss += tmp_eval_loss.item()
      nb_eval_steps += 1

  if nb_eval_steps == 0:
    results = {k: 0 for k in ["loss"]}
  else:
    eval_loss = eval_loss / nb_eval_steps

    results = {
      "loss": eval_loss,
    }

  if print_result:
    logger.info("***** Evaluation result %s in %s *****" % (prefix, lang))
    for key in sorted(results.keys()):
      logger.info("  %s = %s", key, str(results[key]))

  return results, None 


def load_and_cache_examples(args, tokenizer):
  # Make sure only the first process in distributed training process
  # the dataset, and the others will use the cache
  if args.local_rank not in [-1, 0] and not evaluate:
    torch.distributed.barrier()

  # Load data features from cache or dataset file
  cached_features_file = os.path.join(args.data_dir, "cached_pretrain_sde_vocab_ngram{}_{}".format(str(args.max_ngram_size),
    str(args.max_seq_length)))
  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features_vocab = torch.load(cached_features_file)
  else:
    features_vocab = create_sde_pretrain_vocab_features(args.max_seq_length, tokenizer,
                        max_ngram_size=args.max_ngram_size
                        )
    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features_vocab)))
      torch.save(features_vocab, cached_features_file)

  # Make sure only the first process in distributed training process
  # the dataset, and the others will use the cache
  if args.local_rank == 0 and not evaluate:
    torch.distributed.barrier()

  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features_vocab], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features_vocab], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in features_vocab], dtype=torch.long)
  all_label_ids = torch.tensor([f.label_ids for f in features_vocab], dtype=torch.long)
  dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  return dataset


def main():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_dir", default=None, type=str, required=True,
            help="The input data dir. Should contain the training files for the NER/POS task.")
  parser.add_argument("--model_type", default=None, type=str, required=True,
            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
  parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
  parser.add_argument("--output_dir", default=None, type=str, required=True,
            help="The output directory where the model predictions and checkpoints will be written.")

  ## SDE parameters
  parser.add_argument("--max_ngram_size", default=10, type=int,
            help="ngram size for each word")
  parser.add_argument("--bpe_segment", type=int, default=1, help="whether to segment by BPE or by word")
  parser.add_argument("--sde_latent", type=int, default=5000, help="sde latent emb size")

  ## Other parameters
  parser.add_argument("--labels", default="", type=str,
            help="Path to a file containing all labels. If not specified, NER/POS labels are used.")
  parser.add_argument("--config_name", default="", type=str,
            help="Pretrained config name or path if not the same as model_name")
  parser.add_argument("--tokenizer_name", default="", type=str,
            help="Pretrained tokenizer name or path if not the same as model_name")
  parser.add_argument("--cache_dir", default=None, type=str,
            help="Where do you want to store the pre-trained models downloaded from s3")
  parser.add_argument("--max_seq_length", default=128, type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
               "than this will be truncated, sequences shorter will be padded.")
  parser.add_argument("--do_train", action="store_true",
            help="Whether to run training.")
  parser.add_argument("--do_eval", action="store_true",
            help="Whether to run eval on the dev set.")
  parser.add_argument("--do_predict", action="store_true",
            help="Whether to run predictions on the test set.")
  parser.add_argument("--do_predict_dev", action="store_true",
            help="Whether to run predictions on the dev set.")
  parser.add_argument("--init_checkpoint", default=None, type=str,
            help="initial checkpoint for train/predict")
  parser.add_argument("--evaluate_during_training", action="store_true",
            help="Whether to run evaluation during training at each logging step.")
  parser.add_argument("--do_lower_case", action="store_true",
            help="Set this flag if you are using an uncased model.")
  parser.add_argument("--few_shot", default=-1, type=int,
            help="num of few-shot exampes")

  parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
            help="Batch size per GPU/CPU for training.")
  parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
            help="Batch size per GPU/CPU for evaluation.")
  parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument("--learning_rate", default=5e-5, type=float,
            help="The initial learning rate for Adam.")
  parser.add_argument("--weight_decay", default=0.0, type=float,
            help="Weight decay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float,
            help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float,
            help="Max gradient norm.")
  parser.add_argument("--num_train_epochs", default=3.0, type=float,
            help="Total number of training epochs to perform.")
  parser.add_argument("--max_steps", default=-1, type=int,
            help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
  parser.add_argument("--warmup_steps", default=0, type=int,
            help="Linear warmup over warmup_steps.")

  parser.add_argument("--logging_steps", type=int, default=50,
            help="Log every X updates steps.")
  parser.add_argument("--save_steps", type=int, default=50,
            help="Save checkpoint every X updates steps.")
  parser.add_argument("--save_only_best_checkpoint", action="store_true",
            help="Save only the best checkpoint during training")
  parser.add_argument("--eval_all_checkpoints", action="store_true",
            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
  parser.add_argument("--no_cuda", action="store_true",
            help="Avoid using CUDA when available")
  parser.add_argument("--overwrite_output_dir", action="store_true",
            help="Overwrite the content of the output directory")
  parser.add_argument("--overwrite_cache", action="store_true",
            help="Overwrite the cached training and evaluation sets")
  parser.add_argument("--seed", type=int, default=42,
            help="random seed for initialization")

  parser.add_argument("--fp16", action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
  parser.add_argument("--fp16_opt_level", type=str, default="O1",
            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
               "See details at https://nvidia.github.io/apex/amp.html")
  parser.add_argument("--local_rank", type=int, default=-1,
            help="For distributed training: local_rank")
  parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
  parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
  parser.add_argument("--predict_langs", type=str, default="en", help="prediction languages")
  parser.add_argument("--train_langs", default="en", type=str,
            help="The languages in the training sets.")
  parser.add_argument("--log_file", type=str, default=None, help="log file")
  parser.add_argument("--eval_patience", type=int, default=-1, help="wait N times of decreasing dev score before early stop during training")
  args = parser.parse_args()

  if os.path.exists(args.output_dir) and os.listdir(
      args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
      "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
        args.output_dir))

  # Setup distant debugging if needed
  if args.server_ip and args.server_port:
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
  else:
  # Initializes the distributed backend which sychronizes nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device

  # Setup logging
  logging.basicConfig(handlers = [logging.FileHandler(args.log_file), logging.StreamHandler()],
                      format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt = '%m/%d/%Y %H:%M:%S',
                      level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
  logging.info("Input args: %r" % args)
  logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
           args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

  # Set seed
  set_seed(args)

  # Prepare NER/POS task
  labels = get_labels(args.labels)
  num_labels = len(labels)
  # Use cross entropy ignore index as padding label id
  # so that only real label ids contribute to the loss later
  pad_token_label_id = CrossEntropyLoss().ignore_index

  # Load pretrained model and tokenizer
  # Make sure only the first process in distributed training loads model/vocab
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()

  args.model_type = args.model_type.lower()
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                        do_lower_case=args.do_lower_case,
                        cache_dir=args.cache_dir if args.cache_dir else None)
  config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                      num_labels=num_labels,
                      max_ngram_size=args.max_ngram_size,
                      cls_token_id=tokenizer.cls_token_id,
                      sep_token_id=tokenizer.sep_token_id,
                      unk_token_id=tokenizer.unk_token_id,
                      pad_token_id=tokenizer.pad_token_id,
                      cache_dir=args.cache_dir if args.cache_dir else None)

  sde_config_class, sde_model_class, sde_tokenizer_class = MODEL_CLASSES[args.model_type]
  sde_config = sde_config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                      num_labels=num_labels,
                      use_sde_embed=True,
                      sde_latent=args.sde_latent,
                      max_ngram_size=args.max_ngram_size,
                      cls_token_id=tokenizer.cls_token_id,
                      sep_token_id=tokenizer.sep_token_id,
                      unk_token_id=tokenizer.unk_token_id,
                      pad_token_id=tokenizer.pad_token_id,
                      cache_dir=args.cache_dir if args.cache_dir else None)


  if args.init_checkpoint:
    logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
    model = model_class.from_pretrained(args.init_checkpoint,
                                        config=config,
                                        cache_dir=args.init_checkpoint)
    sde_model = sde_model_class.from_pretrained(args.init_checkpoint,
                                        config=sde_config,
                                        cache_dir=args.init_checkpoint)
  else:
    logger.info("loading from cached model = {}".format(args.model_name_or_path))
    model = model_class.from_pretrained(args.model_name_or_path,
                      from_tf=bool(".ckpt" in args.model_name_or_path),
                      config=config,
                      cache_dir=args.cache_dir if args.cache_dir else None)
    sde_model = sde_model_class.from_pretrained(args.model_name_or_path,
                      from_tf=bool(".ckpt" in args.model_name_or_path),
                      config=sde_config,
                      cache_dir=args.cache_dir if args.cache_dir else None)

  lang2id = config.lang2id if args.model_type == "xlm" else None
  logger.info("Using lang2id = {}".format(lang2id))

  # Make sure only the first process in distributed training loads model/vocab
  if args.local_rank == 0:
    torch.distributed.barrier()
  model.requires_grad = False
  model.to(args.device)
  sde_model.to(args.device)
  logger.info("Training/evaluation parameters %s", args)

  # Training
  if args.do_train:
    train_dataset = load_and_cache_examples(args, tokenizer)
    global_step, tr_loss = train(args, train_dataset, sde_model, model, tokenizer, labels, pad_token_label_id, lang2id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

  # Saving best-practices: if you use default names for the model,
  # you can reload it using from_pretrained()
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

    # Save model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    logger.info("Saving model checkpoint to %s", args.output_dir)
    model_to_save = sde_model.module if hasattr(sde_model, "module") else sde_model
    model_to_save.save_pretrained(args.output_dir)
    sde_tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

  # Initialization for evaluation
  results = {}
  if args.init_checkpoint:
    best_checkpoint = args.init_checkpoint
  elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
    best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
  else:
    best_checkpoint = args.output_dir
  best_f1 = 0

if __name__ == "__main__":
  main()
