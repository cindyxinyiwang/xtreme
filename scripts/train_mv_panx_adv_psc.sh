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
##SBATCH --partition=GPU-AI  
#SBATCH --nodes=1                                                                
##SBATCH --gres=gpu:volta16:1                                                             
#SBATCH --gres=gpu:1                                                             
#SBATCH --time=48:00:00

REPO=$PWD
FILE=/ocean/projects/dbs200003p/xinyiw1/
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$FILE/download/"}
OUT_DIR=${4:-"$FILE/outputs/"}

#export CUDA_VISIBLE_DEVICES=$GPU
TASK='panx'
LANGS="ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu"
TRAIN_LANGS="en"
NUM_EPOCHS=10
MAX_LENGTH=128
OPTIM='Adam'
LR=2e-5
BPE_DROP=0.3
SBPED=0
VTAU=0
DTAU=0

RWEIGHT=0.5
DWEIGHT=0.5 
RESAMPLE=0
KL=1

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
ASTEP=2
ANORM=1e-2
AMAG=1e-2
AT="kl"

DATA_DIR=$DATA_DIR/${TASK}/${TASK}_processed_maxlen${MAX_LENGTH}/

for SEED in 1 2 3 4 5;
do
#OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}-TrainLang${TRAIN_LANGS}_sbped${SBPED}_kl${KL}_s${SEED}/"
OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}-TrainLang${TRAIN_LANGS}_mbped${BPE_DROP}_adv_lr${ALR}_as${ASTEP}_an${ANORM}_am${AMAG}_at${AT}_kl${KL}_s${SEED}/"

mkdir -p $OUTPUT_DIR
python $REPO/third_party/run_mv_tag_adv.py \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --per_gpu_eval_batch_size 32 \
  --save_steps 1000 \
  --seed $SEED \
  --learning_rate $LR \
  --do_predict \
  --predict_langs $LANGS \
  --train_langs $TRAIN_LANGS \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --eval_patience -1 \
  --overwrite_output_dir \
  --bpe_dropout $BPE_DROP \
  --kl_weight $KL \
  --drop_tau $DTAU \
  --adv-lr $ALR \
  --adv-steps $ASTEP \
  --adv-max-norm $ANORM \
  --adv-init-mag $AMAG \
  --adv_type $AT \
  --vocab_dist_tau $VTAU \
  --vocab_dist_filename /ocean/projects/dbs200003p/xinyiw1/outputs/bert_${TASK}_${TRAIN_LANGS}.json \
  --save_only_best_checkpoint $LC
done
