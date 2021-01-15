#!/bin/bash
# Copyright 2020 Google and DeepMind.
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
#SBATCH --partition=GPU-AI  
#SBATCH --nodes=1                                                                
#SBATCH --gres=gpu:volta16:1                                                             
#SBATCH --time=48:00:00

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$SCRATCH/download/"}
OUT_DIR=${4:-"$SCRATCH/outputs/"}
#TRAIN_LANGS="is"
TRAIN_LANGS="hi"

if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi
TASK=udpos
MAX_LENGTH=128

DATA_DIR=$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAX_LENGTH}/

python $REPO/third_party/get_vocab_distance.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --data_dir $DATA_DIR \
  --labels $DATA_DIR/labels.txt \
  --train_langs $TRAIN_LANGS \
  --max_seq_length $MAX_LENGTH \
  --output_file $OUT_DIR/${MODEL_TYPE}_${TRAIN_LANGS}".json" 

