#!/bin/bash
sh cluster_experiments.sh Eval eval_rcnn.py $@ --eval_mode rcnn --eval --time 30:00 --set RPN.LOAD_RPN_ONLY False
