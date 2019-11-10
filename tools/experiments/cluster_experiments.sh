#/bin/bash
refine_num_layers=(1 3 5 10 20 30 40)
#refine_num_layers=(1 3 5 10)
#refine_num_layers=(1)
#deepgcn_num_layers=(1 3 5 10 20 30 40)
deepgcn_num_layers=(1 3 5 10)
#deepgcn_num_layers=(1 3)
#deepgcn_num_layers=(3)
#sh cluster_rcnn.sh default "$@"
#sh cluster_rcnn.sh pretrained "$@" --pretrained_rpn
#sh cluster_rcnn.sh rotnet "$@"
#sh cluster_rcnn.sh deepgcn "$@" --gres gpu:v100 --time 40:00:00
#sh cluster_rcnn.sh pretrained_deepgcn "$@" --pretrained_rpn --gres gpu:v100 --time 40:00:00
#sh cluster_rcnn.sh edgeconv "$@"
#sh cluster_rcnn.sh nofine "$@" --custom_rpn nofine
#sh cluster_rcnn.sh refine "$@"
#for refine_num_layer in "${refine_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_refine edge_modilation_$refine_num_layer "$@" --time 60:00:00 --gres gpu:v100 --set RCNN.REF_CONFIG.N_BLOCKS $refine_num_layer --set RCNN.REF_CONFIG.CONV edge --set RCNN.REF_CONFIG.LINEAR_DILATION False; done
#for refine_num_layer in "${refine_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_refine edge_prop_$refine_num_layer "$@" --time 60:00:00 --gres gpu:v100 --set RCNN.REF_CONFIG.N_BLOCKS $refine_num_layer --set RCNN.REF_CONFIG.CONV edge --set RCNN.REF_CONFIG.USE_PROPOSALS True; done
#for refine_num_layer in "${refine_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_refine edge_$refine_num_layer "$@" --time 60:00:00 --gres gpu:v100 --set RCNN.REF_CONFIG.N_BLOCKS $refine_num_layer --set RCNN.REF_CONFIG.CONV edge; done
#for refine_num_layer in "${refine_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_refine $refine_num_layer "$@" --time 60:00:00 --gres gpu:v100 --set RCNN.REF_CONFIG.N_BLOCKS $refine_num_layer; done
#for refine_num_layer in "${refine_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_refine feats_$refine_num_layer "$@" --time 60:00:00 --gres gpu:v100 --set RCNN.REF_CONFIG.USE_RCNN_FEATS True --set RCNN.REF_CONFIG.N_BLOCKS $refine_num_layer; done
for refine_num_layer in "${refine_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_refine edge_feats_$refine_num_layer "$@" --time 60:00:00 --gres gpu:v100 --set RCNN.REF_CONFIG.CONV edge --set RCNN.REF_CONFIG.USE_RCNN_FEATS True --set RCNN.REF_CONFIG.N_BLOCKS $refine_num_layer; done
#for deepgcn_num_layer in "${deepgcn_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_deepgcn edge_$deepgcn_num_layer "$@" --time 200:00:00 --gres gpu:v100 --set RCNN.DEEPGCN_CONFIG.N_BLOCKS $deepgcn_num_layer --set RCNN.DEEPGCN_CONFIG.CONV edge; done
#for deepgcn_num_layer in "${deepgcn_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_deepgcn $deepgcn_num_layer "$@" --time 50:00:00 --gres gpu:v100 --set RCNN.DEEPGCN_CONFIG.N_BLOCKS $deepgcn_num_layer; done
#for deepgcn_num_layer in "${deepgcn_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_deepfeats edge_$deepgcn_num_layer "$@" --time 200:00:00 --gres gpu:v100 --set RCNN.DEEPGCN_CONFIG.N_BLOCKS $deepgcn_num_layer --set RCNN.DEEPGCN_CONFIG.CONV edge; done
#for deepgcn_num_layer in "${deepgcn_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_deepfeats edge_modilation_$deepgcn_num_layer "$@" --time 200:00:00 --gres gpu:v100 --set RCNN.DEEPGCN_CONFIG.N_BLOCKS $deepgcn_num_layer --set RCNN.DEEPGCN_CONFIG.CONV edge --set RCNN.REF_CONFIG.LINEAR_DILATION False; done
#for deepgcn_num_layer in "${deepgcn_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_deepfeats $deepgcn_num_layer "$@" --time 200:00:00 --gres gpu:v100 --set RCNN.DEEPGCN_CONFIG.N_BLOCKS $deepgcn_num_layer; done
#sh cluster_rcnn.sh pretrained_offline default $@ --pretrained_rpn --time 20:00:00
#sh cluster_rcnn.sh pretrained_offline_refine "$@" --pretrained_rpn --time 50:00:00
#sh cluster_rcnn.sh pretrained_offline_deeprefine "20_20" "$@" --pretrained_rpn --gres gpu:v100 --time 50:00:00
#sh cluster_rcnn.sh pretrained_offline_deeprefine "3_3" "$@" --pretrained_rpn --gres gpu:v100 --time 50:00:00 --set RCNN.DEEPGCN_CONFIG.N_BLOCKS 3 --set RCNN.REF_CONFIG.N_BLOCKS 3
#sh cluster_rcnn.sh pretrained_offline_deeprefine "5_20" "$@" --pretrained_rpn --gres gpu:v100 --time 80:00:00 --set RCNN.REF_CONFIG.N_BLOCKS 5
#sh cluster_rcnn.sh pretrained_offline_deeprefine "5_edge_5" "$@" --pretrained_rpn --gres gpu:v100 --time 200:00:00 --set RCNN.DEEPGCN_CONFIG.N_BLOCKS 5 --set RCNN.REF_CONFIG.N_BLOCKS 5 --set RCNN.
#sh cluster_rcnn.sh pretrained_offline_deeprefine "5_20" "$@" --pretrained_rpn --gres gpu:v100 --time 200:00:00 --set RCNN.REF_CONFIG.N_BLOCKS 5
#sh cluster_rcnn.sh pretrained_deeprefine "3_3" "$@" --pretrained_rpn --gres gpu:v100 --time 200:00:00 --set RCNN.DEEPGCN_CONFIG.N_BLOCKS 3 --set RCNN.REF_CONFIG.N_BLOCKS 3
#sh cluster_rcnn.sh pretrained_offline_deepgcn "$@" --pretrained_rpn --time 150:00:00
#for refine_num_layer in "${refine_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_refine $refine_num_layer "$@" --pretrained_rpn --time 50:00:00 --set RCNN.REF_CONFIG.N_BLOCKS $refine_num_layer; done
#for deepgcn_num_layer in "${deepgcn_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_deepgcn $deepgcn_num_layer "$@" --pretrained_rpn --time 200:00:00 --gres gpu:v100 --set RCNN.DEEPGCN_CONFIG.N_BLOCKS $deepgcn_num_layer; done
#for deepgcn_num_layer in "${deepgcn_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_deepgcn constant_dilation_${deepgcn_num_layer} "$@" --pretrained_rpn --time 20:00:00 --gres gpu:v100 --set RCNN.DEEPGCN_CONFIG.CONSTANT_DILATION True --set RCNN.DEEPGCN_CONFIG.N_BLOCKS $deepgcn_num_layer; done
#for deepgcn_num_layer in "${deepgcn_num_layers[@]}"; do sh cluster_rcnn.sh pretrained_offline_deepgcn ${deepgcn_num_layer}_constant_dilation "$@" --time 200:00:00 --gres gpu:v100 --set RCNN.DEEPGCN_CONFIG.CONSTANT_DILATION True --set RCNN.DEEPGCN_CONFIG.N_BLOCKS $deepgcn_num_layer; done
