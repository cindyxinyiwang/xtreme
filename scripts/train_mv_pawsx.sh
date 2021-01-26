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
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=1

TASK='pawsx'
LR=2e-5
EPOCH=5
MAXL=128
LANGS="de,en,es,fr,ja,ko,zh"
BPE_DROP=0
SBPED=0.5
SBPE_END=0
IADV=0.8

WS=0
KL=1
KL_T=1

DTAU=0

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

for SEED in 1 2 3 4 5;
do
#SAVE_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}_sbped${SBPED}_intau${INTERP_TAU}_vtau${VTAU}_kl${KL}_klt${KL_T}_s${SEED}/"
#SAVE_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}_sbped${SBPED}_int${INTERP}_ws${WS}_kl${KL}_klt${KL_T}_s${SEED}/"
SAVE_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}_sbped${SBPED}_kl${KL}_ia${IADV}_s${SEED}/"
#SAVE_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}_sbped${SBPED}_kl${KL}_dtau${DTAU}_s${SEED}/"
mkdir -p $SAVE_DIR

python $PWD/third_party/run_mv_classify.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --train_language en \
  --task_name $TASK \
  --do_train \
  --do_eval \
  --do_predict \
  --train_split train \
  --test_split test \
  --data_dir $DATA_DIR/$TASK/ \
  --gradient_accumulation_steps $GRAD_ACC \
  --save_steps 200 \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --max_seq_length $MAXL \
  --output_dir $SAVE_DIR \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --log_file 'train.log' \
  --predict_languages $LANGS \
  --eval_langs $LANGS \
  --save_only_best_checkpoint $LC \
  --seed $SEED \
  --bpe_dropout $BPE_DROP \
  --sample_bpe_dropout $SBPED \
  --sample_bpe_dropout_end $SBPE_END \
  --kl_weight $KL \
  --kl_t $KL_T \
  --drop_tau $DTAU \
  --inverse_adv_words_tau $IADV \
  --eval_test_set 
  #--vocab_dist_tau $VTAU \
  #--vocab_dist_filename /ocean/projects/dbs200003p/xinyiw1/outputs/bert_panx_en.json \
done
