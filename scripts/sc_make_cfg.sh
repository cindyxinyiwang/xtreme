#!/bin/bash

## regular
CFG_DIR=cfg/panx/
TEMPLATE_NAME=train_panx_template
mkdir -p $CFG_DIR

for SEED in 2 3 4 5; do
for BPEDROP in 0 0.4 0.6; do
  SCRIPT=${CFG_DIR}/${TEMPLATE_NAME}_s${SEED}_bped${BPEDROP}.sh
  sed "s/SETSEED/$SEED/g; s/SETBPEDROP/$BPEDROP/g" < scripts/${TEMPLATE_NAME}.sh > $SCRIPT
  sbatch $SCRIPT
done   
done

## KL
CFG_DIR=cfg/panx/
TEMPLATE_NAME=train_mv_panx_template
BPEDROP=0.4
KL=0.2
mkdir -p $CFG_DIR

for SEED in 2 3 4 5;
do
  SCRIPT=${CFG_DIR}/${TEMPLATE_NAME}_s${SEED}_bped${BPEDROP}_kl${KL}.sh
  sed "s/SETSEED/$SEED/g; s/SETBPEDROP/$BPEDROP/g; s/SETKL/$KL/g" < scripts/${TEMPLATE_NAME}.sh > $SCRIPT 
  sbatch $SCRIPT 
done
