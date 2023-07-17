#!/usr/bin/env bash
trap "exit" INT
base_cmd="python train.py --data_name digits -sc mnist10k  --name bal"
net_cmd=" --net digit -nc 10 --image_size 32 --trans fcn --pre_epoch 5 "
opt_cmd=" -ni 10000 -vi 250 -bs 512 --lr 0.0001 --val_freq 4 "
hp_cmd=" --bal --cl --clw 0.75 -db kaiming -beta 0.1 --adv_steps 10 -clamp -beta 1.0 --lr_adv 5e-6 --trans_depth 4"

rs_vals=(1 2 3 4 5)

#sc_vals=("mnist_m" "svhn" "usps" "synth")
# this block runs 5 seeds on 5 different gpus parallely.
# you can change this to suit your compute resources
GPU=0


for rs in "${rs_vals[@]}"
  do
    nohup $base_cmd $net_cmd $opt_cmd $hp_cmd -g $GPU -rs $rs > runs/digits_bal_"$rs".out &
    ((GPU=(((GPU+1)%2)+0)))
  done

