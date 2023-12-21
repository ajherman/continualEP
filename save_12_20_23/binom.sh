
#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 8
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

epochs=100
cores=6

# Accumulator neuron experiments
beta=0.2
T1=8
T2=3
hidden_size=256
tau_dynamic=0.2

step=0.02 # Keep fixed
batch_size=20

# for omega in {1,2,4,8,16,1024}
# # for omega in {14,15,17,18,63,65}
# # for omega in {0.5,1,3,4,15,16,63,64,2048}
# do
# cep_dir=compare_cep_omega_"$omega"_tau_"$tau_dynamic"
# skewsym_dir=compare_skewsym_omega_"$omega"_tau_"$tau_dynamic"
# stdp0_dir=compare_stdp0_omega_"$omega"_tau_"$tau_dynamic"
# stdp1_dir=compare_stdp1_omega_"$omega"_tau_"$tau_dynamic"
# stdp2_dir=compare_stdp2_omega_"$omega"_tau_"$tau_dynamic"
# stdp3_dir=compare_stdp3_omega_"$omega"_tau_"$tau_dynamic"
# stdp4_dir=compare_stdp4_omega_"$omega"_tau_"$tau_dynamic"
# stdp5_dir=compare_stdp5_omega_"$omega"_tau_"$tau_dynamic"
#
# mkdir -p $cep_dir
# mkdir -p $skewsym_dir
# mkdir -p $stdp0_dir
# mkdir -p $stdp1_dir
# mkdir -p $stdp2_dir
# mkdir -p $stdp3_dir
# mkdir -p $stdp4_dir
# mkdir -p $stdp5_dir
#
# srun -N 1 -n 1 -c $cores -o "$cep_dir".out --open-mode=append ./main_wrapper.sh --M 1 --spiking --load --use-time-variables --directory $cep_dir --omega $omega --step $step --spike-method binomial --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule cep &
# srun -N 1 -n 1 -c $cores -o "$skewsym_dir".out --open-mode=append ./main_wrapper.sh --M 1 --spiking --load --use-time-variables --directory $skewsym_dir --omega $omega --step $step --spike-method binomial --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $cores -o "$stdp0_dir".out --open-mode=append ./main_wrapper.sh --M 1 --spiking --load --use-time-variables --directory $stdp0_dir --omega $omega --step $step --spike-method binomial --tau-dynamic $tau_dynamic --tau-trace 0.025 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$stdp1_dir".out --open-mode=append ./main_wrapper.sh --M 1 --spiking --load --use-time-variables --directory $stdp1_dir --omega $omega --step $step --spike-method binomial --tau-dynamic $tau_dynamic --tau-trace 0.05 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$stdp2_dir".out --open-mode=append ./main_wrapper.sh --M 1 --spiking --load --use-time-variables --directory $stdp2_dir --omega $omega --step $step --spike-method binomial --tau-dynamic $tau_dynamic --tau-trace 0.1 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$stdp3_dir".out --open-mode=append ./main_wrapper.sh --M 1 --spiking --load --use-time-variables --directory $stdp3_dir --omega $omega --step $step --spike-method binomial --tau-dynamic $tau_dynamic --tau-trace 0.2 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$stdp4_dir".out --open-mode=append ./main_wrapper.sh --M 1 --spiking --load --use-time-variables --directory $stdp4_dir --omega $omega --step $step --spike-method binomial --tau-dynamic $tau_dynamic --tau-trace 0.4 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $cores -o "$stdp5_dir".out --open-mode=append ./main_wrapper.sh --M 1 --spiking --load --use-time-variables --directory $stdp5_dir --omega $omega --step $step --spike-method binomial --tau-dynamic $tau_dynamic --tau-trace 0.8 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# done

nonspiking_dir=nonspiking_skewsym
normal_dir=normal_skewsym
binom_dir=binom_skewsym

mkdir -p $nonspiking_dir
mkdir -p $normal_dir
mkdir -p $binom_dir

srun -N 1 -n 1 -c $cores -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --M 1 --spiking --load --use-time-variables --directory $nonspiking_dir --step $step --spike-method nonspiking --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
srun -N 1 -n 1 -c $cores -o "$normal_dir".out --open-mode=append ./main_wrapper.sh --M 1 --spiking --load --use-time-variables --directory $normal_dir --omega 1e24 --step $step --spike-method normal --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
srun -N 1 -n 1 -c $cores -o "$binom_dir".out --open-mode=append ./main_wrapper.sh --M 1 --spiking --load --use-time-variables --directory $binom_dir --omega 2048 --step $step --spike-method binomial --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
