#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 12
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

epochs=100
#hidden_size=512
cores=8
#batch_size=20

# # One layer
# i=0
# beta=0.2
# #batch_size=20
# T1=8
# T2=3
# tau_dynamic=0.2 # Try 0.04 or something smaller than 0.2
# max_fr=5
#
# for step in {0.5,0.2,0.1,0.05}
# do
#
# nonspiking_cep_dir=nonspiking_cep_"$i"
# nonspiking_skewsym_dir=nonspiking_skewsym_"$i"
# nonspiking_stdp_slow_dir=nonspiking_stdp_slow_"$i"
# nonspiking_stdp_med_dir=nonspiking_stdp_med_"$i"
# nonspiking_stdp_fast_dir=nonspiking_stdp_fast_"$i"
# stdp_slow_dir=stdp_slow_"$i"
# stdp_med_dir=stdp_med_"$i"
# stdp_fast_dir=stdp_fast_"$i"
#
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_skewsym_dir
# mkdir -p $stdp_slow_dir
# mkdir -p $stdp_med_dir
# mkdir -p $stdp_fast_dir
# mkdir -p $nonspiking_stdp_slow_dir
# mkdir -p $nonspiking_stdp_med_dir
# mkdir -p $nonspiking_stdp_fast_dir
# # Added in spiking!
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_cep_dir --spiking --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_skewsym_dir --spiking --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_slow_dir --spiking --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_med_dir --spiking --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.5  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_fast_dir --spiking --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.05  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# srun -N 1 -n 1 -c $cores -o "$stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_slow_dir --spiking --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_med_dir --spiking --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.5  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_fast_dir --spiking --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.05  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
# i=$((i+1))
#
# done

# Two layer

# i=0
# beta=0.5
# T1=5
# T2=1
# # tau_dynamic=0.02
# max_fr=1
# # Was 0.2,0.02 for both
# for tau_dynamic in {0.05,0.02,0.01}
# do
# for step in {0.05,0.02,0.01}
# do
#
# nonspiking_cep_dir=nonspiking_cep_"$i"_2layer
# nonspiking_skewsym_dir=nonspiking_skewsym_"$i"_2layer
# #nonspiking_stdp_slow_dir=nonspiking_stdp_slow_"$i"_2layer
# #nonspiking_stdp_med_dir=nonspiking_stdp_med_"$i"_2layer
# #nonspiking_stdp_fast_dir=nonspiking_stdp_fast_"$i"_2layer
# #stdp_slow_dir=stdp_slow_"$i"_2layer
# #stdp_med_dir=stdp_med_"$i"_2layer
# #stdp_fast_dir=stdp_fast_"$i"_2layer
#
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_skewsym_dir
# #mkdir -p $stdp_slow_dir
# #mkdir -p $stdp_med_dir
# #mkdir -p $stdp_fast_dir
# #mkdir -p $nonspiking_stdp_slow_dirs
# #mkdir -p $nonspiking_stdp_med_dir
# #mkdir -p $nonspiking_stdp_fast_dir
#
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_cep_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_skewsym_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# #srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_slow_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.1  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# #srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_med_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.05  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# #srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_fast_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# #srun -N 1 -n 1 -c $cores -o "$stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_slow_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.1  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# #srun -N 1 -n 1 -c $cores -o "$stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_med_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.05  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# #srun -N 1 -n 1 -c $cores -o "$stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_fast_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
# i=$((i+1))
# done
# done

# # Campare hidden sizes
#
# i=0
# beta=0.5
# T1=5
# T2=1
# # tau_dynamic=0.02
# max_fr=1
# # Was 0.2,0.02 for both
# tau_dynamic=0.02
# step=0.05
# for hidden_size1 in {256,512}
# do
# for hidden_size2 in {256,512}
# do
#
# nonspiking_cep_dir=nonspiking_cep_"$i"_2layer
# nonspiking_skewsym_dir=nonspiking_skewsym_"$i"_2layer
# nonspiking_stdp_slow_dir=nonspiking_stdp_slow_"$i"_2layer
# nonspiking_stdp_med_dir=nonspiking_stdp_med_"$i"_2layer
# nonspiking_stdp_fast_dir=nonspiking_stdp_fast_"$i"_2layer
# stdp_slow_dir=stdp_slow_"$i"_2layer
# stdp_med_dir=stdp_med_"$i"_2layer
# stdp_fast_dir=stdp_fast_"$i"_2layer
#
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_skewsym_dir
# mkdir -p $stdp_slow_dir
# mkdir -p $stdp_med_dir
# mkdir -p $stdp_fast_dir
# mkdir -p $nonspiking_stdp_slow_dir
# mkdir -p $nonspiking_stdp_med_dir
# mkdir -p $nonspiking_stdp_fast_dir
#
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_cep_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_skewsym_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_slow_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.1  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_med_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.05  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_fast_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# srun -N 1 -n 1 -c $cores -o "$stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_slow_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.1  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_med_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.05  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_fast_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
# i=$((i+1))
# done
# done


# One layer spiking dynamics

# i=0
# beta=0.5
# T1=5
# T2=1
# # tau_dynamic=0.02
# max_fr=1
# # Was 0.2,0.02 for both
# hidden_size=256
# tau_dynamic=0.02
# step=0.05
# for batch_size in {20,50,100,200}
# do
# for step in {0.05,0.02,0.01,0.005}
# do
#
# nonspiking_cep_dir=nonspiking_cep_"$i"_2layer
# nonspiking_skewsym_dir=nonspiking_skewsym_"$i"_2layer
# # nonspiking_stdp_slow_dir=nonspiking_stdp_slow_"$i"_2layer
# # nonspiking_stdp_med_dir=nonspiking_stdp_med_"$i"_2layer
# # nonspiking_stdp_fast_dir=nonspiking_stdp_fast_"$i"_2layer
# # stdp_slow_dir=stdp_slow_"$i"_2layer
# # stdp_med_dir=stdp_med_"$i"_2layer
# # stdp_fast_dir=stdp_fast_"$i"_2layer
#
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_skewsym_dir
# # mkdir -p $stdp_slow_dir
# # mkdir -p $stdp_med_dir
# # mkdir -p $stdp_fast_dir
# # mkdir -p $nonspiking_stdp_slow_dir
# # mkdir -p $nonspiking_stdp_med_dir
# # mkdir -p $nonspiking_stdp_fast_dir
#
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $nonspiking_cep_dir --step $step --spike-method accumulator --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $nonspiking_skewsym_dir --step $step --spike-method accumulator --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# # srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $nonspiking_stdp_slow_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.1  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# # srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $nonspiking_stdp_med_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.05  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# # srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $nonspiking_stdp_fast_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# # srun -N 1 -n 1 -c $cores -o "$stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $stdp_slow_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.1  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# # srun -N 1 -n 1 -c $cores -o "$stdp_med_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $stdp_med_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.05  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# # srun -N 1 -n 1 -c $cores -o "$stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $stdp_fast_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size1 $hidden_size2 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
# i=$((i+1))
# done
# done

# Accumulator test
# beta=0.5
# T1=5
# T2=1
# hidden_size=256
# tau_dynamic=0.02
# step=0.05

beta=0.2
T1=8
T2=3
hidden_size=256
tau_dynamic=0.2
step=0.05
batch_size=20

nonspiking_cep_poi_dir=nonspiking_cep_poisson_"$batch_size"
nonspiking_skewsym_poi_dir=nonspiking_skewsym_poisson_"$batch_size"
nonspiking_stdp_poi_dir=nonspiking_stdp_poisson_"$batch_size"
spiking_stdp_poi_dir=spiking_stdp_poisson_"$batch_size"

mkdir -p $nonspiking_cep_poi_dir
mkdir -p $nonspiking_skewsym_poi_dir
mkdir -p $nonspiking_stdp_poi_dir
mkdir -p $spiking_stdp_poi_dir

srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_poi_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $nonspiking_cep_poi_dir --step $step --spike-method poisson --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule cep &
srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_poi_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $nonspiking_skewsym_poi_dir --step $step --spike-method poisson --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_poi_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $nonspiking_stdp_poi_dir --step $step --spike-method poisson --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
srun -N 1 -n 1 -c $cores -o "$spiking_stdp_poi_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $spiking_stdp_poi_dir --step $step --spike-method poisson --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &


for omega in {1,2,4,2048}
do

nonspiking_cep_acc_dir=nonspiking_cep_accumulator_"$omega"_"$batch_size"
nonspiking_skewsym_acc_dir=nonspiking_skewsym_accumulator_"$omega"_"$batch_size"
nonspiking_stdp_acc_dir=nonspiking_stdp_accumulator_"$omega"_"$batch_size"
spiking_stdp_acc_dir=spiking_stdp_accumulator_"$omega"_"$batch_size"

mkdir -p $nonspiking_cep_acc_dir
mkdir -p $nonspiking_skewsym_acc_dir
mkdir -p $nonspiking_stdp_acc_dir
mkdir -p $spiking_stdp_acc_dir

srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_acc_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $nonspiking_cep_acc_dir --omega $omega --step $step --spike-method accumulator --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule cep &
srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_acc_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $nonspiking_skewsym_acc_dir --omega $omega --step $step --spike-method accumulator --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_acc_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $nonspiking_stdp_acc_dir --step $step --spike-method accumulator --omega $omega --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
srun -N 1 -n 1 -c $cores -o "$spiking_stdp_acc_dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $spiking_stdp_acc_dir --step $step --spike-method accumulator --omega $omega --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
done


# Compare difference methods of spike generation

beta=0.2
T1=8
T2=3
hidden_size=256
tau_dynamic=0.2
step=0.025
batch_size=200
tau_trace=0.5

i=0
for update_rule in {'cep','skewsym','nonspikingstdp','stdp'}
do
for spike_method in {'none','poisson'}
do
dir=compare_spike_methods_"$update_rule"_"$spike_method"
i=$((i+1))
mkdir -p $dir
srun -N 1 -n 1 -c $cores -o "$dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $dir --step $step --spike-method $spike_method --tau-dynamic $tau_dynamic --tau-trace $tau_trace --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule $update_rule &
done
spike_method='accumulator'
for omega in {1,2,4}
do
dir=compare_spike_methods_"$update_rule"_"$spike_method"
i=$((i+1))
mkdir -p $dir
srun -N 1 -n 1 -c $cores -o "$dir".out --open-mode=append ./main_wrapper.sh --spiking --load --use-time-variables --directory $dir --omega $omega --step $step --spike-method accumulator --tau-dynamic $tau_dynamic --tau-trace $tau_trace --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule $update_rule &
done
done

dir=compare_spike_method_"$i"
