#!/bin/bash
sh cluster_experiments.sh Offline_Test eval_rcnn.py --eval_mode rcnn_offline --rcnn_eval_roi_dir /ibex/scratch/zarzarj/PointRCNN/output/pretrained_offline_test/eval/epoch_no_number/test/test_mode/detections/data --rcnn_eval_feature_dir /ibex/scratch/zarzarj/PointRCNN/output/pretrained_offline_test/eval/epoch_no_number/test/test_mode/features --set RPN.LOAD_RPN_ONLY False --set TEST.SPLIT test --time 5:00:00 --test --eval $@
