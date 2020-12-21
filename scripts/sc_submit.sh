#!/bin/bash

#SBATCH --partition=GPU-AI  
#SBATCH --nodes=1                                                                
#SBATCH --gres=gpu:volta16:1                                                             
#SBATCH --time=8:00:00


CFG_DIR=cfg/xnli/
mkdir -p $CFG_DIR

for f in `ls $CFG_DIR | grep -v .started$`; do
    if [[ ! -e $CFG_DIR/$f.started ]]; then
        echo "running $f"
        touch $CFG_DIR/$f.started
        hostname
        nvidia-smi
        bash $CFG_DIR/$f
    else
        echo "already started $f"
    fi
done
