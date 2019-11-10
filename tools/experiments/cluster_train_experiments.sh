#!/bin/bash
sh cluster_experiments.sh Train train_rcnn.py --train_mode rcnn --epochs 70 --ckpt_save_interval 2 --rpn --batch_size 4
