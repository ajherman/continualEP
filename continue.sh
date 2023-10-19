#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 16
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

cores=4

for dir in "$1"/*
do
if [ -d "$dir" ]
then
echo $dir
srun -N 1 -n 1 -c $cores -o "$dir".out --open-mode=append ./main_wrapper.sh --load --directory $dir &
fi
done
