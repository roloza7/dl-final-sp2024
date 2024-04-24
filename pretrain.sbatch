#!/bin/bash
#SBATCH -JPreTrainMAE
#SBATCH -N2 --gres=gpu:4 --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH -t 1:00:00
#SBATCH --output Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rloza3@gatech.edu
#SBATCH --signal=10@120
cd $SLURM_SUBMIT_DIR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module load anaconda3/2023.03
conda activate dl-final-sp2024
srun python train.py --imroot coco/images --annfile coco/annotations/ann2017/captions_train2017.json --save-path checkpoints/MAE-L --num-workers 4 --epochs 200 --batch-size 512 --parallel --load-path checkpoints/MAE-L
