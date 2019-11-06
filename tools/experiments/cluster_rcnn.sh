#!/bin/bash
config="$1"
shift
exp_name="$1"
shift
suffix="$1"
shift
output_dir="/ibex/scratch/zarzarj/PointRCNN/output"
#output_dir="../output"
SLURMARGS=()
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --qos|--time|--gres)
    SLURMARGS+=("$1")
    shift # past argument
    SLURMARGS+=("$1")
    shift # past value
    ;;
    --eval)
    shift
    POSITIONAL+=(" --ckpt ${output_dir}/rcnn_${config}_${exp_name}/ckpt/checkpoint_epoch_"$1".pth")
    shift
    ;;
    --eval_all)
    POSITIONAL+=(" --eval_all --ckpt_dir ${output_dir}/rcnn_${config}_${exp_name}/ckpt/")
    shift
    ;;
    --resume)
    shift
    POSITIONAL+=(" --ckpt ${output_dir}/rcnn_${config}_${exp_name}/ckpt/checkpoint_epoch_"$1".pth")
    shift
    ;;
    --rpn)
    shift
    if [ "$1" == --custom_rpn ]; then
        shift
        POSITIONAL+=" --rpn_ckpt ../output/rpn_"$1"/ckpt/checkpoint_epoch_155.pth"
        shift
    else
	if [ "$1" == --pretrained_rpn ]; then
	    shift
            POSITIONAL+=" --rpn_ckpt ../output/PointRCNN.pth"
        else
            POSITIONAL+=" --rpn_ckpt ../output/rpn/default/ckpt/checkpoint_epoch_155.pth"
        fi
    fi
    ;;
    --custom_rpn)
    shift
    shift
    ;;
    --pretrained_rpn)
    shift
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters
sbatch "${SLURMARGS[@]}" <<EOT
#!/bin/bash
#SBATCH -J ${config}_${suffix}_${exp_name}
#SBATCH -o %x.%A.out
#SBATCH -e %x.%A.err
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
##SBATCH --reservation=IVUL
#SBATCH --time=20:00:00

# activate your conda env
echo "Loading anaconda..."

module purge
module load anaconda3

source activate pointrcnn
conda deactivate
source activate pointrcnn
echo "...Anaconda env loaded"

#run the training:
echo "Starting python function ..."
cd ..
python $@ --cfg_file cfgs/${config}.yaml --output_dir ${output_dir}/rcnn_${config}_${exp_name}/
echo "...training function Done"

EOT
