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
#SBATCH --partition=GPU-shared  
#SBATCH --nodes=1                                                                
#SBATCH --gres=gpu:1                                                             
#SBATCH --time=48:00:00

REPO=$PWD
#MODEL=${1:-bert-base-multilingual-cased}
MODEL=${1:-xlm-roberta-base}
GPU=${2:-0}
FILE=/ocean/projects/dbs200003p/xinyiw1/
DATA_DIR=${3:-"$FILE/download/"}
OUT_DIR=${4:-"$FILE/outputs/"}

TASK='udpos'
#LANGS='af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh'
#TRAIN_LANGS="en"
#TRAIN_LANGS="is"
#LANGS="is,fo"
#TRAIN_LANGS="fi"
#LANGS="fi,olo"

#TRAIN_LANGS="hi"
#LANGS="hi,bho,ur"

TRAIN_LANGS="pt"
LANGS="gl,pt"

NUM_EPOCHS=10
MAX_LENGTH=128
LR=2e-5
BPE_DROP=0.2
KL=0.2 
KL_T=1

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
  BATCH_SIZE=8
  GRAD_ACC=4
fi

ALR=1e-3
ASTEP=1
ANORM=1e-5
AMAG=1e-5

TAU=0
DTAU=0
VTAU=1

DATA_DIR=$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAX_LENGTH}/
for SEED in 2 3;
do
OUTPUT_DIR="$OUT_DIR/${TASK}_${TRAIN_LANGS}/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}_mbped${BPE_DROP}_vtau${VTAU}_dtau${DTAU}_wadv_lr${ALR}_as${ASTEP}_an${ANORM}_am${AMAG}_kl${KL}_s${SEED}/"
mkdir -p $OUTPUT_DIR
python $REPO/third_party/run_mv_tag_wadv.py \
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
  --bpe_dropout $BPE_DROP \
  --kl_weight $KL \
  --kl_t $KL_T \
  --adv-lr $ALR \
  --adv-steps $ASTEP \
  --adv-max-norm $ANORM \
  --adv-init-mag $AMAG \
  --tau $TAU \
  --drop_tau $DTAU \
  --vocab_dist_filename /pylon5/dbs200003p/xinyiw1/outputs/bert_${TRAIN_LANGS}.json \
  --vocab_dist_tau $VTAU \
  --save_only_best_checkpoint $LC
done