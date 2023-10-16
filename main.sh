#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 12
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

epochs=100
hidden_size=512 #256
cores=10
batch_size=20

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
#
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_cep_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_skewsym_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_slow_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_med_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.5  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_fast_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.05  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# srun -N 1 -n 1 -c $cores -o "$stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_slow_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_med_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.5  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_fast_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.05  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
# i=$((i+1))
#
# done

# Two layer

i=0
beta=0.5
T1=4
T2=1
# tau_dynamic=0.02
max_fr=1
# Was 0.2,0.02 for both
for tau_dynamic in {0.01,0.02,0.005}
do
for step in {0.05,0.1,0.2}
do

nonspiking_cep_dir=nonspiking_cep_"$i"_2layer
nonspiking_skewsym_dir=nonspiking_skewsym_"$i"_2layer
nonspiking_stdp_slow_dir=nonspiking_stdp_slow_"$i"_2layer
nonspiking_stdp_med_dir=nonspiking_stdp_med_"$i"_2layer
nonspiking_stdp_fast_dir=nonspiking_stdp_fast_"$i"_2layer
stdp_slow_dir=stdp_slow_"$i"_2layer
stdp_med_dir=stdp_med_"$i"_2layer
stdp_fast_dir=stdp_fast_"$i"_2layer

mkdir -p $nonspiking_cep_dir
mkdir -p $nonspiking_skewsym_dir
mkdir -p $stdp_slow_dir
mkdir -p $stdp_med_dir
mkdir -p $stdp_fast_dir
mkdir -p $nonspiking_stdp_slow_dir
mkdir -p $nonspiking_stdp_med_dir
mkdir -p $nonspiking_stdp_fast_dir

srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_cep_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule cep &
srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_skewsym_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_slow_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_med_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.01  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $nonspiking_stdp_fast_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.005  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
srun -N 1 -n 1 -c $cores -o "$stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_slow_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.02  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
srun -N 1 -n 1 -c $cores -o "$stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_med_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.01  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
srun -N 1 -n 1 -c $cores -o "$stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --use-time-variables --directory $stdp_fast_dir --step $step --max-fr $max_fr --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 0.005  --activation-function hardsigm --size_tab 10 $hidden_size $hidden_size 784 --lr_tab 0.00018 0.0018 0.01 --epochs $epochs --T1 $T1  --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &

i=$((i+1))
done
done



# beta=0.2
# N1=40
# N2=15
# for n_trace in {0.03,0.3,3.0}
# do
# for n_dynamic in {0.2,0.5,1.0,2.0}
# do
# # spiking_cep_dir=spiking_cep_old_"$i"
# nonspiking_cep_dir=nonspiking_cep_"$((i/4))"_"$((i%4))"
# # nonspiking_cepalt_dir=nonspiking_cepalt_old_"$i"
# nonspiking_skewsym_dir=nonspiking_skewsym_"$((i/4))"_"$((i%4))"
# # stdp_dir=stdp_old_"$i"
# nonspiking_stdp_dir=nonspiking_stdp_"$((i/4))"_"$((i%4))"
# stdp_dir=stdp_"$((i/4))"_"$((i%4))"
#
# # mkdir -p $spiking_cep_dir
# mkdir -p $nonspiking_cep_dir
# # mkdir -p $nonspiking_cepalt_dir
# mkdir -p $nonspiking_skewsym_dir
# # mkdir -p $stdp_dir
# mkdir -p $stdp_dir
# mkdir -p $nonspiking_stdp_dir
#
# # srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cep_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# # srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# # srun -N 1 -n 1 -c $cores -o "$stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $stdp_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_stdp_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --n-trace $n_trace  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule nonspikingstdp &
# srun -N 1 -n 1 -c $cores -o "$stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $stdp_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --n-trace $n_trace  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# i=$((i+1))
# done
# done

#epochs=150
#hidden_size=256
#cores=10
#batch_size=25
#N1=40
#N2=15
#beta=0.2
#i=0
#for n_dynamic in {1,2,3,4,5,6,7,8}
#do
#  spiking_cep_dir=spiking_cep_old_tau="$i"
#  nonspiking_cep_dir=nonspiking_cep_old_tau="$i"
#  nonspiking_cepalt_dir=nonspiking_cepalt_old_tau="$i"
#  nonspiking_skewsym_dir=nonspiking_skewsym_old_tau="$i"
  # stdp_dir=stdp_old_"$i"

#  mkdir -p $spiking_cep_dir
#  mkdir -p $nonspiking_cep_dir
#  mkdir -p $nonspiking_cepalt_dir
#  mkdir -p $nonspiking_skewsym_dir
  # mkdir -p $stdp_dir

#  srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
#  srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cep_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
#  srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
#  srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
#  # srun -N 1 -n 1 -c $cores -o "$stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $stdp_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#  i=$((i+1))
#done



#
#
#
#
#










# # Spiking and nonspiking cepalt for various levels of discretization
# #######################################################################################################################################################
# i=0
# for N2 in {3,6,9,12}
# 	do
# 		N1=$((3*N2))
# 		beta=0.2
# 		update_rule="cepalt"
# 		spiking_dir=cepalt_spiking_"$i"
# 		nonspiking_dir=cepalt_nonspiking_"$i"
# 		mkdir -p $spiking_dir
# 		mkdir -p $nonspiking_dir
# 		srun -N 1 -n 1 -c 6 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
# 		srun -N 1 -n 1 -c 6 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
# 		i=$((i+1))
# 	done
#
# for N2 in {15,18,21,24}
# 	do
# 		N1=$((3*N2))
# 		beta=0.2
# 		update_rule="cepalt"
# 		spiking_dir=cepalt_spiking_"$i"
# 		nonspiking_dir=cepalt_nonspiking_"$i"
# 		mkdir -p $spiking_dir
# 		mkdir -p $nonspiking_dir
# 		srun -N 1 -n 1 -c 9 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
# 		srun -N 1 -n 1 -c 9 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
# 		i=$((i+1))
# 	done
#
# for N2 in {27,30}
# 	do
# 		N1=$((3*N2))
# 		beta=0.2
# 		update_rule="cepalt"
# 		spiking_dir=cepalt_spiking_"$i"
# 		nonspiking_dir=cepalt_nonspiking_"$i"
# 		mkdir -p $spiking_dir
# 		mkdir -p $nonspiking_dir
# 		srun -N 1 -n 1 -c 12 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
# 		srun -N 1 -n 1 -c 12 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
# 		i=$((i+1))
# 	done
# ######################################################################################################################################################

# # Plot comparing spiking/nonspiking and different learning rule types
# ########################################################################################################################################################
# i=0
# for batch_size in {25,50,100,200}
#  	do
#  		beta=1.0
# 		N1=40
# 		N2=15
#  		skewsym_spiking_dir=spiking_skewsym_"$i"
#  		skewsym_nonspiking_dir=nonspiking_skewsym_"$i"
#  		spiking_stdp_slow_dir=spiking_stdp_slow_"$i"
# 		spiking_stdp_med_dir=spiking_stdp_med_"$i"
# 		spiking_stdp_fast_dir=spiking_stdp_fast_"$i"
# 		nonspiking_stdp_slow_dir=nonspiking_stdp_slow_"$i"
# 		nonspiking_stdp_med_dir=nonspiking_stdp_med_"$i"
# 		nonspiking_stdp_fast_dir=nonspiking_stdp_fast_"$i"
#  		mkdir -p $skewsym_spiking_dir
#  		mkdir -p $skewsym_nonspiking_dir
#  		mkdir -p $spiking_stdp_slow_dir
# 		mkdir -p $spiking_stdp_med_dir
# 		mkdir -p $spiking_stdp_fast_dir
# 		mkdir -p $nonspiking_stdp_slow_dir
# 		mkdir -p $nonspiking_stdp_med_dir
# 		mkdir -p $nonspiking_stdp_fast_dir
# 		srun -N 1 -n 1 -c 4 -o "$skewsym_spiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $skewsym_spiking_dir --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# 		srun -N 1 -n 1 -c 4 -o "$spiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_slow_dir --spiking --action train --batch-size $batch_size --tau-trace 2.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 		srun -N 1 -n 1 -c 4 -o "$spiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_med_dir --spiking --action train --batch-size $batch_size --tau-trace 1.44  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 		srun -N 1 -n 1 -c 4 -o "$spiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_fast_dir --spiking --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 		srun -N 1 -n 1 -c 4 -o "$skewsym_nonspiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $skewsym_nonspiking_dir --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# 		srun -N 1 -n 1 -c 4 -o "$nonspiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_stdp_slow_dir --action train --batch-size $batch_size --tau-trace 2.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 		srun -N 1 -n 1 -c 4 -o "$nonspiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_stdp_med_dir --action train --batch-size $batch_size --tau-trace 1.44  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 		srun -N 1 -n 1 -c 4 -o "$nonspiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_stdp_fast_dir --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#   	i=$((i+1))
#  	done
# ########################################################################################################################################################################################


# # Compare spiking/nonspiking cepalt/skewsym
# ######################################################################################################################################################
# N1=40
# N2=15
# beta=0.5
# step=1.0
# max_fr=1.0
# spiking_skewsym_dir=spiking_skewsym
# nonspiking_skewsym_dir=nonspiking_skewsym
# spiking_cepalt_dir=spiking_cepalt
# nonspiking_cepalt_dir=nonspiking_cepalt
#
# mkdir -p $spiking_skewsym_dir
# mkdir -p $nonspiking_skewsym_dir
# mkdir -p $spiking_cepalt_dir
# mkdir -p $nonspiking_cepalt_dir
#
# srun -N 1 -n 1 -c 6 -o "$spiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_skewsym_dir --spike-height 1.0 --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 50 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c 6 -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_skewsym_dir --spike-height 1.0 --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 50 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c 6 -o "$spiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_cepalt_dir --spike-height 1.0 --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 50 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# srun -N 1 -n 1 -c 6 -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_cepalt_dir --spike-height 1.0 --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 50 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# ######################################################################################################################################################


# # Spiking and nonspiking cepalt for various levels of discretization (fixed max_fr)
# #######################################################################################################################################################
# i=0
# for N2 in {3,6,9,12}
# 	do
# 		N1=$((3*N2))
#     max_fr=3.0
# 		beta=0.2
# 		update_rule="cepalt"
# 		spiking_dir=cepalt_spiking_"$i"
# 		nonspiking_dir=cepalt_nonspiking_"$i"
# 		mkdir -p $spiking_dir
# 		mkdir -p $nonspiking_dir
# 		srun -N 1 -n 1 -c 6 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_dir --max-fr $max_fr --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule $update_rule &
# 		srun -N 1 -n 1 -c 6 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_dir --max-fr $max_fr --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule $update_rule  &
# 		i=$((i+1))
# 	done
#
# for N2 in {15,18,21,24}
# 	do
# 		N1=$((3*N2))
# 		beta=0.2
# 		update_rule="cepalt"
# 		spiking_dir=cepalt_spiking_"$i"
# 		nonspiking_dir=cepalt_nonspiking_"$i"
# 		mkdir -p $spiking_dir
# 		mkdir -p $nonspiking_dir
# 		srun -N 1 -n 1 -c 9 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_dir --max-fr $max_fr --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule $update_rule &
# 		srun -N 1 -n 1 -c 9 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_dir --max-fr $max_fr --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule $update_rule  &
# 		i=$((i+1))
# 	done
#
# for N2 in {27,30,33,36}
# 	do
# 		N1=$((3*N2))
# 		beta=0.2
# 		update_rule="cepalt"
# 		spiking_dir=cepalt_spiking_"$i"
# 		nonspiking_dir=cepalt_nonspiking_"$i"
# 		mkdir -p $spiking_dir
# 		mkdir -p $nonspiking_dir
# 		srun -N 1 -n 1 -c 12 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_dir --max-fr $max_fr --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule $update_rule &
# 		srun -N 1 -n 1 -c 12 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_dir --max-fr $max_fr --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule $update_rule  &
# 		i=$((i+1))
# 	done
# ######################################################################################################################################################

# # Universal
# #########################################################################################################################################
# epochs=100
# beta=0.2 #0.2,0.5,0.9
# tau_dynamic=3 #2,3,4,5
# max_q=5 #3,5,10,20
# batch_size=25 #25, 50,100,200
# for N2 in {15,25}
# do
#   ten_beta=$(echo "scale=2; 10*$beta" | bc)
#   ten_beta=${ten_beta%.*}
#   N1=$((3*N2))
#   cores=$((N2/3))
#   # Spiking networks
#   spiking_cepalt_dir=spiking_cepalt_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   spiking_skewsym_dir=spiking_skewsym_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   spiking_stdp_slow_dir=spiking_stdp_slow_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   spiking_stdp_med_dir=spiking_stdp_med_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   spiking_stdp_fast_dir=spiking_stdp_fast_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   mkdir -p $spiking_cepalt_dir
#   mkdir -p $spiking_skewsym_dir
#   mkdir -p $spiking_stdp_slow_dir
#   mkdir -p $spiking_stdp_med_dir
#   mkdir -p $spiking_stdp_fast_dir
#   srun -N 1 -n 1 -c $cores -o "$spiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cepalt_dir --tau-dynamic $tau_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# 	srun -N 1 -n 1 -c $cores -o "$spiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_skewsym_dir --tau-dynamic $tau_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# 	srun -N 1 -n 1 -c $cores -o "$spiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_slow_dir --tau-dynamic $tau_dynamic --spiking --action train --batch-size $batch_size --tau-trace 2.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 	srun -N 1 -n 1 -c $cores -o "$spiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_med_dir --tau-dynamic $tau_dynamic --spiking --action train --batch-size $batch_size --tau-trace 1.5  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 	srun -N 1 -n 1 -c $cores -o "$spiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_fast_dir --tau-dynamic $tau_dynamic --spiking --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
#   #Nonspiking networks
#   nonspiking_cepalt_dir=nonspiking_cepalt_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   nonspiking_skewsym_dir=nonspiking_skewsym_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   nonspiking_stdp_slow_dir=nonspiking_stdp_slow_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   nonspiking_stdp_med_dir=nonspiking_stdp_med_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   nonspiking_stdp_fast_dir=nonspiking_stdp_fast_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   mkdir -p $nonspiking_cepalt_dir
#   mkdir -p $nonspiking_skewsym_dir
#   mkdir -p $nonspiking_stdp_slow_dir
#   mkdir -p $nonspiking_stdp_med_dir
#   mkdir -p $nonspiking_stdp_fast_dir
#   srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
#   srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# 	srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_stdp_slow_dir --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 2.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 	srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_stdp_med_dir --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 1.5  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 	srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_stdp_fast_dir --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# done


# # Universal
# #########################################################################################################################################
# epochs=300
# # beta=0.2 #0.2,0.5,0.9
# # n_dynamic=3.5 #2,3,4,5
# # max_Q=5 #3,5,10,20
# # Use N2 nodes
# # batch_size=200 #25, 50,100,200
# n_dynamic=8
# N1=$((8*n_dynamic))
# N2=$((8*n_dynamic))
# for batch_size in {200,25}
# do
#   for beta in {1,2,4,8}
#   do
#     ten_beta=$(echo "scale=2; 10*$beta" | bc)
#     ten_beta=${ten_beta%.*}
#     # N1=$((3*N2))
#     cores=10
#     # Spiking networks
#     spiking_cep_dir=cep_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     spiking_skew_dir=skew_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     spiking_cepalt_dir=cepalt_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     spiking_skewsym_dir=skewsym_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     spiking_stdp_0_dir=stdp_0_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     spiking_stdp_1_dir=stdp_1_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     spiking_stdp_2_dir=stdp_2_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     spiking_stdp_3_dir=stdp_3_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     mkdir -p $spiking_cep_dir
#     mkdir -p $spiking_skew_dir
#     mkdir -p $spiking_cepalt_dir
#     mkdir -p $spiking_skewsym_dir
#     mkdir -p $spiking_stdp_0_dir
#     mkdir -p $spiking_stdp_1_dir
#     mkdir -p $spiking_stdp_2_dir
#     mkdir -p $spiking_stdp_3_dir
#     srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
#   	#srun -N 1 -n 1 -c $cores -o "$spiking_skew_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_skew_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skew &
#     #srun -N 1 -n 1 -c $cores -o "$spiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cepalt_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
#   	srun -N 1 -n 1 -c $cores -o "$spiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_skewsym_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
#     srun -N 1 -n 1 -c $cores -o "$spiking_stdp_0_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_0_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 2.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#     srun -N 1 -n 1 -c $cores -o "$spiking_stdp_1_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_1_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#     srun -N 1 -n 1 -c $cores -o "$spiking_stdp_2_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_2_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 6.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#   	srun -N 1 -n 1 -c $cores -o "$spiking_stdp_3_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_3_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 8.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
#
#     # #Nonspiking networks
#     # nonspiking_cepalt_dir=nonspiking_cepalt_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     # nonspiking_skewsym_dir=nonspiking_skewsym_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     # nonspiking_stdp_slow_dir=nonspiking_stdp_slow_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     # nonspiking_stdp_med_dir=nonspiking_stdp_med_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     # nonspiking_stdp_fast_dir=nonspiking_stdp_fast_maxfr="$max_fr"_N2="$N2"_tau="$tau_dynamic"_beta="$ten_beta"_batch="$batch_size"
#     # mkdir -p $nonspiking_cepalt_dir
#     # mkdir -p $nonspiking_skewsym_dir
#     # mkdir -p $nonspiking_stdp_slow_dir
#     # mkdir -p $nonspiking_stdp_med_dir
#     # mkdir -p $nonspiking_stdp_fast_dir
#     # srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
#     # srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
#   	# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_stdp_slow_dir --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 2.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#   	# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_stdp_med_dir --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 1.5  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#   	# srun -N 1 -n 1 -c $cores -o "$nonspiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_stdp_fast_dir --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#   done
# done


# Final
#########################################################################################################################################
#
# # Original params
# epochs=300
# hidden_size=256
# cores=12
#
# n_dynamic=3
# beta=0.5
# N1=40
# N2=15
# batch_size=25
#
# spiking_cep_dir=spiking_cep_old_A
# nonspiking_cep_dir=nonspiking_cep_old_A
# nonspiking_cepalt_dir=nonspiking_cepalt_old_A
# nonspiking_skewsym_dir=nonspiking_skewsym_old_A
# stdp_dir=stdp_old_A
#
# mkdir -p $spiking_cep_dir
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_cepalt_dir
# mkdir -p $nonspiking_skewsym_dir
# mkdir -p $stdp_dir
#
# srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cep_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $stdp_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
#
#
# n_dynamic=4
# beta=0.5
# N1=40
# N2=15
# batch_size=25
#
# spiking_cep_dir=spiking_cep_old_B
# nonspiking_cep_dir=nonspiking_cep_old_B
# nonspiking_cepalt_dir=nonspiking_cepalt_old_B
# nonspiking_skewsym_dir=nonspiking_skewsym_old_B
# stdp_dir=stdp_old_B
#
# mkdir -p $spiking_cep_dir
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_cepalt_dir
# mkdir -p $nonspiking_skewsym_dir
# mkdir -p $stdp_dir
#
# srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cep_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $stdp_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
#
#
#
# n_dynamic=5
# beta=0.5
# N1=40
# N2=15
# batch_size=25
#
# spiking_cep_dir=spiking_cep_old_C
# nonspiking_cep_dir=nonspiking_cep_old_C
# nonspiking_cepalt_dir=nonspiking_cepalt_old_C
# nonspiking_skewsym_dir=nonspiking_skewsym_old_C
# stdp_dir=stdp_old_C
#
# mkdir -p $spiking_cep_dir
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_cepalt_dir
# mkdir -p $nonspiking_skewsym_dir
# mkdir -p $stdp_dir
#
# srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cep_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $stdp_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
#
#
# n_dynamic=3
# beta=1.5
# N1=40
# N2=15
# batch_size=25
#
# spiking_cep_dir=spiking_cep_old_D
# nonspiking_cep_dir=nonspiking_cep_old_D
# nonspiking_cepalt_dir=nonspiking_cepalt_old_D
# nonspiking_skewsym_dir=nonspiking_skewsym_old_D
# stdp_dir=stdp_old_D
#
# mkdir -p $spiking_cep_dir
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_cepalt_dir
# mkdir -p $nonspiking_skewsym_dir
# mkdir -p $stdp_dir
#
# srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cep_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $stdp_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
#
#
# n_dynamic=4
# beta=1.5
# N1=40
# N2=15
# batch_size=25
#
# spiking_cep_dir=spiking_cep_old_E
# nonspiking_cep_dir=nonspiking_cep_old_E
# nonspiking_cepalt_dir=nonspiking_cepalt_old_E
# nonspiking_skewsym_dir=nonspiking_skewsym_old_E
# stdp_dir=stdp_old_E
#
# mkdir -p $spiking_cep_dir
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_cepalt_dir
# mkdir -p $nonspiking_skewsym_dir
# mkdir -p $stdp_dir
#
# srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cep_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $stdp_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
#
# n_dynamic=7
# beta=1.5
# N1=40
# N2=15
# batch_size=25
#
# spiking_cep_dir=spiking_cep_old_F
# nonspiking_cep_dir=nonspiking_cep_old_F
# nonspiking_cepalt_dir=nonspiking_cepalt_old_F
# nonspiking_skewsym_dir=nonspiking_skewsym_old_F
# stdp_dir=stdp_old_F
#
# mkdir -p $spiking_cep_dir
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_cepalt_dir
# mkdir -p $nonspiking_skewsym_dir
# mkdir -p $stdp_dir
#
# srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cep_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $stdp_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
# ##########################################################33
#
# # Original params
# epochs=300
# hidden_size=256
# cores=12
#
# n_dynamic=6 #8
# beta=0.9 #1.0
# N1=72 #40
# N2=24 #15
# batch_size=200 #25 #200
#
# spiking_cep_dir=spiking_cep_old
# nonspiking_cep_dir=nonspiking_cep_old
# nonspiking_cepalt_dir=nonspiking_cepalt_old
# nonspiking_skewsym_dir=nonspiking_skewsym_old
# stdp_dir=stdp_old
#
# mkdir -p $spiking_cep_dir
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_cepalt_dir
# mkdir -p $nonspiking_skewsym_dir
# mkdir -p $stdp_dir
#
# srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cep_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $stdp_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
#
#
# # # # New params
# #
# n_dynamic=8
# beta=2
# N1=64
# N2=64
# batch_size=200
# #
# spiking_cep_dir=spiking_cep_new
# nonspiking_cep_dir=nonspiking_cep_new
# nonspiking_cepalt_dir=nonspiking_cepalt_new
# nonspiking_skewsym_dir=nonspiking_skewsym_new
# stdp_dir=stdp_new
# #
# mkdir -p $spiking_cep_dir
# mkdir -p $nonspiking_cep_dir
# mkdir -p $nonspiking_cepalt_dir
# mkdir -p $nonspiking_skewsym_dir
# mkdir -p $stdp_dir
# #
# srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cep_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_cepalt_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# srun -N 1 -n 1 -c $cores -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_skewsym_dir --n-dynamic $n_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$stdp_dir".out --open-mode=append ./main_wrapper.sh --load --directory $stdp_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# ###################################################################################################################################3
#

# epochs=300
# # beta=0.2 #0.2,0.5,0.9
# # n_dynamic=3.5 #2,3,4,5
# # max_Q=5 #3,5,10,20
# # Use N2 nodes
# # batch_size=200 #25, 50,100,200
# n_dynamic=8
# batch_size=200
# beta=2
# N1=$((8*n_dynamic))
# N2=$((8*n_dynamic))
#
# ten_beta=$(echo "scale=2; 10*$beta" | bc)
# ten_beta=${ten_beta%.*}
# # N1=$((3*N2))
# cores=10
# # Spiking networks
# spiking_cep_dir=cep_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
# spiking_skew_dir=skew_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
# spiking_cepalt_dir=cepalt_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
# spiking_skewsym_dir=skewsym_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
# spiking_stdp_0_dir=stdp_0_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
# spiking_stdp_1_dir=stdp_1_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
# spiking_stdp_2_dir=stdp_2_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
# spiking_stdp_3_dir=stdp_3_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
# mkdir -p $spiking_cep_dir
# mkdir -p $spiking_skew_dir
# mkdir -p $spiking_cepalt_dir
# mkdir -p $spiking_skewsym_dir
# mkdir -p $spiking_stdp_0_dir
# mkdir -p $spiking_stdp_1_dir
# mkdir -p $spiking_stdp_2_dir
# mkdir -p $spiking_stdp_3_dir
# srun -N 1 -n 1 -c $cores -o "$spiking_cep_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cep_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# #srun -N 1 -n 1 -c $cores -o "$spiking_skew_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_skew_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skew &
# #srun -N 1 -n 1 -c $cores -o "$spiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cepalt_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# srun -N 1 -n 1 -c $cores -o "$spiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_skewsym_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$spiking_stdp_0_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_0_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 2.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$spiking_stdp_1_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_1_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$spiking_stdp_2_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_2_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 6.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$spiking_stdp_3_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_3_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 8.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#
# ############################################################################################
#



# # Test N1
# #########################################################################################################################################
# epochs=100
# batch_size=200 #25, 50,100,200
# n_dynamic=8
# N1=$((8*n_dynamic))
# beta=1
# for k in {3,4,5,6,7,8}
#   do
#   ten_beta=$(echo "scale=2; 10*$beta" | bc)
#   ten_beta=${ten_beta%.*}
#   N2=$((k*n_dynamic))
#   cores=$((2*(k+1)))
#   # Spiking networks
#   # spiking_cepalt_dir=cepalt_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   # spiking_skewsym_dir=skewsym_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   # spiking_stdp_rock_dir=stdp_rock_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   spiking_stdp_slug_dir=stdp_slug_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   spiking_stdp_slow_dir=stdp_slow_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   spiking_stdp_med_dir=stdp_med_N1="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   # spiking_stdp_fast_dir=stdp_fast_N1_="$N1"_N2="$N2"_dyn="$n_dynamic"_beta="$ten_beta"_batch="$batch_size"
#   # mkdir -p $spiking_cepalt_dir
#   # mkdir -p $spiking_skewsym_dir
#   # mkdir -p $spiking_stdp_rock_dir
#   mkdir -p $spiking_stdp_slug_dir
#   mkdir -p $spiking_stdp_slow_dir
#   mkdir -p $spiking_stdp_med_dir
#   # mkdir -p $spiking_stdp_fast_dir
#   # srun -N 1 -n 1 -c $cores -o "$spiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_cepalt_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule cepalt &
# 	# srun -N 1 -n 1 -c $cores -o "$spiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_skewsym_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
#   # srun -N 1 -n 1 -c $cores -o "$spiking_stdp_rock_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_rock_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 5.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#   srun -N 1 -n 1 -c $cores -o "$spiking_stdp_slug_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_slug_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 4.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#   srun -N 1 -n 1 -c $cores -o "$spiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_slow_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 3.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1  --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 	srun -N 1 -n 1 -c $cores -o "$spiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_med_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 2.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
#   # srun -N 1 -n 1 -c $cores -o "$spiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_fast_dir --n-dynamic $n_dynamic --spiking --action train --batch-size $batch_size --n-trace 1.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs $epochs --N1 $N1 --N2 $N2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# done




	# i=0
	# for dt in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
	# 	do
	# 		beta=0.2
	# 		spiking_directory=skewsym_spiking_"$i"
	# 		nonspiking_directory=skewsym_nonspiking_"$i"
	# 		stdp_directory=stdp_"$i"
	# 		mkdir -p $spiking_directory
	# 		mkdir -p $nonspiking_directory
	# 		mkdir -p $stdp_directory
	# 		nohup python -u main.py --directory $spiking_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 40 --N2 15 --beta 0.2 --dt $dt --cep --learning-rule stdp --update-rule skewsym >> skewsym_spiking_log_"$i".out &
	# 		nohup python -u main.py --directory $stdp_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 40 --N2 15 --beta 0.2 --dt $dt --cep --learning-rule stdp --update-rule stdp >> stdp_log_"$i".out &
	# 		nohup python -u main.py --directory $nonspiking_directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 40 --N2 15 --beta 0.2 --dt $dt --cep --learning-rule stdp --update-rule skewsym >> skewsym_nonspiking_log_"$i".out &
	# 		i=$((i+1))
	# 	done

# i=0
# for dt in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
# 	do
# 	beta=0.2
# 	update_rule="cepalt"
# 	directory=cepalt_"$i"
# 	mkdir -p $directory
# 	nohup python -u main.py --directory $directory --load True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 40 --N2 15 --beta 0.2 --dt $dt --cep --learning-rule stdp --update-rule cepalt >> log_"$i".out &
# 	i=$((i+1))
# 	done

# beta=0.2
# dt=0.5
# update_rule="skewsym"
# directory='skewsym_dt=05'
# mkdir $directory
# nohup python -u main.py --directory $directory --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 40 --N2 15 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule $update_rule >> '$directory'/log.out &
#
# beta=0.2
# dt=0.2
# update_rule="skewsym"
# directory='skewsym_dt=02'
# mkdir $directory
# nohup python -u main.py --directory $directory --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 40 --N2 15 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule $update_rule >> '$directory'/log.out


## Fast
#for update_rule in {skewsym,asym,cepalt}
#  do
#    i=0
#    for beta in {0.2,0.8}
#    do
#      for dt in {0.2,0.5,1.0}
#      do
#        echo rule = $update_rule , beta = $beta , dt = $dt > fast_"$update_rule"_"$i".out
#        echo "" >> fast_"$update_rule"_"$i".out
#        nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 40 --N2 15 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule $update_rule >> fast_"$update_rule"_"$i".out &
#        sleep .01
#	i=$((i+1))
#      done
#    done
#  done

# # Slow
# for update_rule in {skewsym,asym,cepalt}
#   do
#     i=0
#     for beta in {0.2,0.5,1.0}
#     do
#       for dt in {0.2,0.5,1.0}
#       do
#         echo rule = $update_rule , beta = $beta , dt = $dt >> nonspike_slow_"$i".out
#         echo "" >> nonspike_slow_"$i".out
#         nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 80 --N2 30 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule  skewsym >> slow_"$update_rule"_"$i".out &
#         i=$((i+1))
#       done
#     done
#   done

# # STDP
# i=0
# for beta in {0.1,0.2,0.5,0.8,1.0}
# do
#   for dt in {0.1,0.2,0.5,1.0}
#   do
#     echo beta = $beta , dt = $dt >> stdp_"$i".out
#     echo "" >> stdp_"$i".out
#     nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.005 --epochs 30 --N1 50 --N2 25 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule stdp >> stdp_"$i".out &
#     i=$((i+1))
#   done
# done

# # Asym
# i=0
# for beta in {0.1,0.2,0.5,0.8,1.0}
# do
#   for dt in {0.1,0.2,0.5,1.0}
#   do
#     echo beta = $beta , dt = $dt >> asym_"$i".out
#     echo "" >> asym_"$i".out
#     nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --N1 40 --N2 15 --beta $beta --dt $dt --learning-rule 'vf' --update-rule 'asym2' --cep >> asym_"$i".out &
#     i=$((i+1))
#   done
# done
#
# # N2
# for N2 in {15,20,25,30,35,40}
#   do
#     nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --N1 100 --N2 $N2 --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  stdp > stdp_N2_"$N2".out &
#   done
#
# # N2
# for N2 in {15,20,25,30,35,40}
#   do
#   nohup python -u main.py --action train --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --N1 80 --N2 $N2 --beta 0.2 --dt 0.2 --cep > cont_N2_"$N2".out &
#   done
