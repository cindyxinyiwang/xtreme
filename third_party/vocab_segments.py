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
"""Fine-tuning models for NER and POS tagging."""

from __future__ import absolute_import, division, print_function
import json
import argparse
import glob
import logging
import os
import random

import pandas as pd
import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils_tag import convert_examples_to_features
from utils_tag import get_labels
from utils_tag import read_examples_from_file
from RecAdam import RecAdam, anneal_function
import utils

from transformers import (
  AdamW,
  get_linear_schedule_with_warmup,
  WEIGHTS_NAME,
  BertConfig,
  BertTokenizer,
  BertForTokenClassification,
  BertForMLMandTokenClassification,
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
  "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
  "bert_mlm": (BertConfig, BertForMLMandTokenClassification, BertTokenizer),
  "xlm": (XLMConfig, XLMForTokenClassification, XLMTokenizer),
  "xlmr": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

def read_to_tokenized_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, lang2id=None):
  # Make sure only the first process in distributed training process
  # the dataset, and the others will use the cache
  if args.local_rank not in [-1, 0] and not evaluate:
    torch.distributed.barrier()

  # Load data features from cache or dataset file
  bpe_dropout = args.bpe_dropout
  langs = lang.split(',')
  logger.info("all languages = {}".format(lang))
  features = []
  data_file = os.path.join(args.data_dir, lang, "{}.{}".format(mode, args.model_name_or_path))
  logger.info("Creating features from dataset file at {} in language {}".format(data_file, lang))
  examples = read_examples_from_file(data_file, lang, lang2id)
  #for (ex_index, example) in enumerate(examples):
  #  if ex_index % 10000 == 0:
  #    logger.info("Writing example %d of %d", ex_index, len(examples))

  #  tokens = []
  #  label_ids = []
  #  for word, label in zip(example.words, example.labels):
  #    if isinstance(tokenizer, XLMTokenizer):
  #      word_tokens = tokenizer.tokenize(word, lang=lang, dropout=bpe_dropout)
  #    else:
  #      word_tokens = tokenizer.tokenize(word, dropout=bpe_dropout)
  #    if len(word) != 0 and len(word_tokens) == 0:
  #      word_tokens = [tokenizer.unk_token]
  #    tokens.append(word_tokens)
  #  example.words = tokens
  return examples

def bucket_by_segment(examples, prediction_file, buckets):
  predictions = []
  trg = []
  print(prediction_file)
  with open(prediction_file, 'r') as myfile:
    for line in myfile:
      line = line.strip()
      if not line:
        predictions.append(trg)
        trg = []
      else:
        trg.append(line)
  if trg: predictions.append(trg)
  print(len(examples), len(predictions))
  #buckets = [[], [], [], [], []]
  for ex, trg in zip(examples, predictions):
    print(ex.words)
    print(ex.labels)
    print(trg)
    print()
    exit(0)
    if len(ex.labels) != len(trg):
      print(ex.words, ex.labels, trg)
      continue
    for word, label, predict in zip(ex.words, ex.labels, trg):
      if len(word) == 1:
        buckets[0].append([word, label, predict])
      elif len(word) == 2:
        buckets[1].append([word, label, predict])
      elif len(word) == 3:
        buckets[2].append([word, label, predict])
      elif len(word) == 4:
        buckets[3].append([word, label, predict])
      else:
        buckets[4].append([word, label, predict])

def bucket_f1(buckets):
  for i in range(5):
    data = buckets[i]
    labels = [d[1] for d in data]
    preds = [d[2] for d in data]
    f1 = f1_score(labels, preds)
    print("bucket {} f1 {}".format(i, f1))

def write_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, lang2id=None):
  # Make sure only the first process in distributed training process
  # the dataset, and the others will use the cache
  if args.local_rank not in [-1, 0] and not evaluate:
    torch.distributed.barrier()

  # Load data features from cache or dataset file
  bpe_dropout = args.bpe_dropout
  if bpe_dropout > 0:
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}_drop{}".format(mode, lang,
      list(filter(None, args.model_name_or_path.split("/"))).pop(),
      str(args.max_seq_length), bpe_dropout))
  else:
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, lang,
      list(filter(None, args.model_name_or_path.split("/"))).pop(),
      str(args.max_seq_length)))
  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
  else:
    langs = lang.split(',')
    logger.info("all languages = {}".format(lang))
    features = []
    data_file = os.path.join(args.data_dir, lang, "{}.{}".format(mode, args.model_name_or_path))
    logger.info("Creating features from dataset file at {} in language {}".format(data_file, lang))
    examples = read_examples_from_file(data_file, lang, lang2id)
    features_lg = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                        cls_token_at_end=bool(args.model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                        pad_on_left=bool(args.model_type in ["xlnet"]),
                        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                        pad_token_label_id=pad_token_label_id,
                        lang=lang,
                        bpe_dropout=bpe_dropout,
                        )
    features.extend(features_lg)
    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
      torch.save(features, cached_features_file)
  
  label_map = {i: label for i, label in enumerate(labels)}
  output_text_file =  cached_features_file + ".txt"
  print("writing to {}...".format(output_text_file))
  outfile = open(output_text_file, 'w')
  examples = []
  for f in features:
    input_ids = f.input_ids
    label_ids = f.label_ids
    input_text = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
    labels = [label_map[i] for i in label_ids if i > 0]
    outfile.write(" ".join(input_text) + '\n')
    outfile.write(" ".join(labels) + '\n')
    examples.append([input_text, labels])
  return examples

def ave_token_segments(examples, tokenizer):
  seg_counts = []
  for ex in examples:
    tokens = ex.words
    for word in tokens:
      subwords = tokenizer.tokenize(word)
      seg_counts.append(len(subwords))
  total_word_count = len(seg_counts)
  seg_counts = np.array(seg_counts)

  #if args.model_type.startswith("xlmr"):
  #  for ex in examples:
  #    tokens = ex[0]
  #    num_segments = 1
  #    for i, t in enumerate(tokens):
  #      if t.startswith("▁"):
  #        seg_counts.append(num_segments)
  #        num_segments = 1
  #      else:
  #        num_segments += 1
  #  total_word_count = len(seg_counts)
  #  seg_counts = np.array(seg_counts)
  #else:
  #  for ex in examples:
  #    tokens = ex[0]
  #    num_segments = 1
  #    for i, t in enumerate(tokens):
  #      if not t.startswith("##"):
  #        seg_counts.append(num_segments)
  #        num_segments = 1
  #      else:
  #        num_segments += 1
  #  total_word_count = len(seg_counts)
  #  seg_counts = np.array(seg_counts)
  return sum(seg_counts)/total_word_count

def sent_len_percent(examples):
  sent_lens = []
  for ex in examples:
    tokens = ex[0]
    sent_lens.append(len(tokens))
  sent_lens = np.array(sent_lens)
  total_count = len(sent_lens)
  print("<10: {}".format(sum(sent_lens < 10) / total_count))
  print("10<=, <20: {}".format(sum(np.logical_and(sent_lens>=10, sent_lens<20)) / total_count))
  print("20<=, <30: {}".format(sum(np.logical_and(sent_lens>=20, sent_lens<30)) / total_count))
  print("30<=, <40: {}".format(sum(np.logical_and(sent_lens>=30, sent_lens<40)) / total_count))
  print("40<=, <50: {}".format(sum(np.logical_and(sent_lens>=40, sent_lens<50)) / total_count))
  print("50<=: {}".format(sum(sent_lens>=50) / total_count))
 

def token_split_percent(examples, tokenizer):
  results = []
  seg_counts = []
  for ex in examples:
    tokens = ex.words
    for word in tokens:
      subwords = tokenizer.tokenize(word)
      seg_counts.append(len(subwords))
  total_word_count = len(seg_counts)
  seg_counts = np.array(seg_counts)
  print("seg 1: {}".format(sum(seg_counts == 1) / total_word_count))
  results.append(sum(seg_counts == 1) / total_word_count)
  print("seg 2: {}".format(sum(seg_counts == 2) / total_word_count))
  results.append(sum(seg_counts == 2) / total_word_count)
  print("seg 3: {}".format(sum(seg_counts == 3) / total_word_count))
  results.append(sum(seg_counts == 3) / total_word_count)
  print("seg 4: {}".format(sum(seg_counts == 4) / total_word_count))
  results.append(sum(seg_counts == 4) / total_word_count)
  print("seg 5: {}".format(sum(seg_counts == 5) / total_word_count))
  results.append(sum(seg_counts == 5) / total_word_count)
  print("seg 6: {}".format(sum(seg_counts == 6) / total_word_count))
  results.append(sum(seg_counts == 6) / total_word_count)
  print("seg 7: {}".format(sum(seg_counts == 7) / total_word_count))
  results.append(sum(seg_counts == 7) / total_word_count)
  print("seg 8: {}".format(sum(seg_counts == 8) / total_word_count))
  results.append(sum(seg_counts == 8) / total_word_count)
  print("seg 9: {}".format(sum(seg_counts == 9) / total_word_count))
  results.append(sum(seg_counts == 9) / total_word_count)
  print("seg >=10: {}".format(sum(seg_counts >= 10) / total_word_count))
  results.append(sum(seg_counts >= 10) / total_word_count)
  return results

def main():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_dir", default=None, type=str, required=True,
            help="The input data dir. Should contain the training files for the NER/POS task.")
  parser.add_argument("--model_type", default=None, type=str, required=True,
            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
  parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))

  parser.add_argument("--output_dir", default=None, type=str)
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
  parser.add_argument("--do_predict_train", action="store_true")
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

  ## SDE parameters
  parser.add_argument("--max_ngram_size", default=10, type=int,
            help="ngram size for each word")
  parser.add_argument("--bpe_segment", type=int, default=1, help="whether to segment by BPE or by word")
  parser.add_argument("--sde_latent", type=int, default=5000, help="sde latent emb size")
  parser.add_argument("--use_sde_embed", action="store_true")
  parser.add_argument("--add_sde_embed", action="store_true")

  parser.add_argument("--tau", type=float, default=-1, help="wait N times of decreasing dev score before early stop during training")

  parser.add_argument("--attention_t", type=float, default=1, help="wait N times of decreasing dev score before early stop during training")
  parser.add_argument("--mlm_weight", type=float, default=-1, help="wait N times of decreasing dev score before early stop during training")
  parser.add_argument("--mlm_lang", type=str, default='ur', help="wait N times of decreasing dev score before early stop during training")
  parser.add_argument("--mlm_start_epoch", type=int, default=0, help="wait N times of decreasing dev score before early stop during training")
  parser.add_argument("--mlm_end_epoch", type=int, default=0, help="wait N times of decreasing dev score before early stop during training")


  parser.add_argument("--update_pretrained_epoch", type=int, default=0, help="wait N times of decreasing dev score before early stop during training")
  parser.add_argument("--bpe_dropout", default=0, type=float)
  parser.add_argument("--resample_dataset", default=0, type=float, help="set to 1 if resample at each epoch")
  parser.add_argument("--fix_class", action='store_true')
  # RecAdam parameters
  parser.add_argument("--optimizer", type=str, default="RecAdam", choices=["Adam", "RecAdam"],
                      help="Choose the optimizer to use. Default RecAdam.")
  parser.add_argument("--recadam_anneal_fun", type=str, default='sigmoid', choices=["sigmoid", "linear", 'constant'],
                      help="the type of annealing function in RecAdam. Default sigmoid")
  parser.add_argument("--recadam_anneal_k", type=float, default=0.5, help="k for the annealing function in RecAdam.")
  parser.add_argument("--recadam_anneal_t0", type=int, default=250, help="t0 for the annealing function in RecAdam.")
  parser.add_argument("--recadam_anneal_w", type=float, default=1.0,
                      help="Weight for the annealing function in RecAdam. Default 1.0.")
  parser.add_argument("--recadam_pretrain_cof", type=float, default=5000.0,
                      help="Coefficient of the quadratic penalty in RecAdam. Default 5000.0.")

  parser.add_argument("--logging_Euclid_dist", action="store_true",
                      help="Whether to log the Euclidean distance between the pretrained model and fine-tuning model")
  parser.add_argument("--start_from_pretrain", action="store_true",
                      help="Whether to initialize the model with pretrained parameters")

  parser.add_argument("--albert_dropout", default=0.0, type=float,
                      help="The dropout rate for the ALBERT model")

  parser.add_argument("--few_shot_extra_langs", type=str, default=None)
  parser.add_argument("--few_shot_extra_langs_size", type=str, default=None)
  args = parser.parse_args()


  # Prepare NER/POS task
  labels = get_labels(args.labels)
  num_labels = len(labels)
  # Use cross entropy ignore index as padding label id
  # so that only real label ids contribute to the loss later
  pad_token_label_id = CrossEntropyLoss().ignore_index

  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                      num_labels=num_labels,
                      use_sde_embed=args.use_sde_embed,
                      add_sde_embed=args.add_sde_embed,
                      sde_latent=args.sde_latent,
                      mlm_weight=args.mlm_weight,
                      attention_t=args.attention_t,
                      fix_class=args.fix_class,
                      cache_dir=args.cache_dir if args.cache_dir else None)
  tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                        do_lower_case=args.do_lower_case,
                        cache_dir=args.cache_dir if args.cache_dir else None)

  print(tokenizer.tokenize("excitement"))
  print(tokenizer.tokenize("ordinateur"))
  print(tokenizer.tokenize("excitação"))
  print(tokenizer.tokenize("правительство"))
  #results = {}
  #buckets = [[], [], [], [], []]
  #for lang in args.predict_langs.split(","):
  #    prediction_file = os.path.join(args.output_dir, "test_{}_predictions.txt".format(lang))  
  #    examples = read_to_tokenized_examples(args, tokenizer, labels, pad_token_label_id, mode="test", lang=lang)
  #    print("lang={}".format(lang))
  #    percent = token_split_percent(examples, tokenizer)
  #    results[lang] = percent
  #    #ave_token = ave_token_segments(examples, tokenizer)
  #    #print("lang={} ave_token={}".format(lang, ave_token))
  #    #bucket_by_segment(examples, prediction_file, buckets)
  ##bucket_f1(buckets)
  #with open('lang_seg_percent.json', 'w') as outfile:
  #  json.dump(results, outfile)

if __name__ == "__main__":
  main()
