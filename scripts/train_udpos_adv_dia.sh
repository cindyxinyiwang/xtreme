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
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=1
TASK='udpos'
#LANGS='af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh'
#TRAIN_LANGS="en"
TRAIN_LANGS="el"
LANGS="el,grc"
DIA_LANGS="grc"
NUM_EPOCHS=10
MAX_LENGTH=128
LR=2e-5

BPE_DROP=0.2
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=4
  GRAD_ACC=8
fi

ALR=3e-2
ASTEP=2
ANORM=1.6e-1
AMAG=1.4e-1

DATA_DIR=$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAX_LENGTH}/
for SEED in 1;
do
OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}_bped${BPE_DROP}_adv_dia_lr${ALR}_as${ASTEP}_an${ANORM}_am${AMAG}_s${SEED}/"
mkdir -p $OUTPUT_DIR
python $REPO/third_party/run_tag_adv.py \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps 500 \
  --seed $SEED \
  --learning_rate $LR \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_langs $LANGS \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --train_langs $TRAIN_LANGS \
  --adv-lr $ALR \
  --adv-steps $ASTEP \
  --adv-max-norm $ANORM \
  --adv-init-mag $AMAG \
  --bpe_dropout $BPE_DROP \
  --dia_langs $DIA_LANGS \
  --save_only_best_checkpoint $LC
done
