#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 16
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

dirs="$1"/*

for dir in dirs:
do
echo $dir
srun -N 1 -n 1 -c $cores -o "$dir".out --open-mode=append ./main_wrapper.sh --load --directory $dir &

done
