### get the vocab edit distance for mBERT
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
import utils


from transformers import (
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
import json
import editdistance
import argparse 
import os

ALL_MODELS = sum(
  (tuple(conf.pretrained_config_archive_map.keys())
    for conf in (BertConfig, XLMConfig, XLMRobertaConfig)),
  ()
)

MODEL_CLASSES = {
  "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
  "xlmr": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}

def remove_bpe(word, indicator):
  if word.startswith(indicator):
    word = word[len(indicator):]
  return word

def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, lang2id=None, few_shot=-1, few_shot_extra_langs=None, few_shot_extra_langs_size=None, bpe_dropout=0):
  # Load data features from cache or dataset file
  if bpe_dropout > 0:
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}_drop{}".format(mode, lang,
      list(filter(None, args.model_name_or_path.split("/"))).pop(),
      str(args.max_seq_length), bpe_dropout))
  else:
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, lang,
      list(filter(None, args.model_name_or_path.split("/"))).pop(),
      str(args.max_seq_length)))
  if os.path.exists(cached_features_file):
    features = torch.load(cached_features_file)
  else:
    langs = lang.split(',')
    features = []
    for lg in langs:
      data_file = os.path.join(args.data_dir, lg, "{}.{}".format(mode, args.model_name_or_path))
      print("Creating features from dataset file at {} in language {}".format(data_file, lg))
      examples = read_examples_from_file(data_file, lg, lang2id)
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
                          lang=lg,
                          bpe_dropout=bpe_dropout,
                          )
      features.extend(features_lg)

  # Convert to Tensors and build dataset
  input_id_set = set([])
  for f in features:
    input_id_set.update(f.input_ids)
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
  all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
  if args.model_type == 'xlm' and features[0].langs is not None:
    all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
    print('all_langs[0] = {}'.format(all_langs[0]))
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_langs)
  else:
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  return dataset, input_id_set



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
  parser.add_argument("--model_type", default=None, type=str, required=True)
  parser.add_argument("--tokenizer_name", default="", type=str,
            help="Pretrained tokenizer name or path if not the same as model_name")
  parser.add_argument("--cache_dir", default=None, type=str,
            help="Where do you want to store the pre-trained models downloaded from s3")

  parser.add_argument("--do_lower_case", action="store_true",
            help="Set this flag if you are using an uncased model.")
  parser.add_argument("--data_dir", default=None, type=str)
  parser.add_argument("--output_file", default=None, type=str,
            help="Where do you want to save the json vocab file")
  parser.add_argument("--train_langs", default=None, type=str)
  parser.add_argument("--labels", default=None, type=str)
  parser.add_argument("--bpe_dropout", default=0, type=float)
  parser.add_argument("--max_seq_length", default=128, type=int)
  args = parser.parse_args()

  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                        do_lower_case=args.do_lower_case,
                        cache_dir=args.cache_dir if args.cache_dir else None)

  # Prepare NER/POS task
  labels = get_labels(args.labels)
  num_labels = len(labels)
  print(args.model_name_or_path)
  print("bert" in args.model_name_or_path)
  # Use cross entropy ignore index as padding label id
  # so that only real label ids contribute to the loss later
  pad_token_label_id = CrossEntropyLoss().ignore_index
  train_dataset, input_id_set = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train", lang=args.train_langs, bpe_dropout=0.2)
  #train_dataset, input_id_set_drop = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train", lang=args.train_langs, bpe_dropout=0)
  #input_id_set = input_id_set.union(input_id_set_drop)
  #train_dataset, input_id_set_drop = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train", lang=args.train_langs, bpe_dropout=0.1)
  #input_id_set = input_id_set.union(input_id_set_drop)
  #train_dataset, input_id_set_drop = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train", lang=args.train_langs, bpe_dropout=0.3)
  #input_id_set = input_id_set.union(input_id_set_drop)
  vocab_distance = {}
  #input_id_set = [ i for i in list(input_id_set) if i > 104 ]
  input_id_set = [ i for i in list(input_id_set)]
  print(len(input_id_set))
  print(args.output_file)
  for i in input_id_set:
    dist = []
    compare_tokens = []
    if "xlm" not in args.model_name_or_path:
      base_token = tokenizer.ids_to_tokens[i]
    else:
      base_token = tokenizer._convert_id_to_token(i)
    if "xlm" not in args.model_name_or_path:
      indicator = "##"
    else:
      indicator = "##"
    base_token = remove_bpe(base_token, indicator)
    for j in input_id_set:
      if i == j:
        dist.append(1000)
        continue
      if "xlm" not in args.model_name_or_path:
        compare_token = tokenizer.ids_to_tokens[j]
      else:
        compare_token = tokenizer._convert_id_to_token(j)
      compare_token = remove_bpe(compare_token, indicator)
      dist.append(editdistance.eval(base_token, compare_token))
    vocab_distance[i] = dist
    if i%1000 == 0:
      print("processed {} words...".format(i))
  out_data = {"vocab": input_id_set,  "vocab_distance": vocab_distance}
  with open(args.output_file, 'w') as outfile:
    json.dump(out_data, outfile)

if __name__ == "__main__":
  main()
