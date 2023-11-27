
#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 15
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

epochs=100
cores=14

# Accumulator neuron experiments
beta=0.2
T1=8
T2=3
hidden_size=256
step=0.02 # Keep fixed
batch_size=20
tau_dynamic=0.2
for M in {7,}
do
for omega in {1,7,1e24}
do
skewsym_dir=normal_skewsym_M_"$M"_omega_"$omega"
stdp1_dir=normal_stdp1_M_"$M"_omega_"$omega"
stdp2_dir=normal_stdp2_M_"$M"_omega_"$omega"
stdp3_dir=normal_stdp3_M_"$M"_omega_"$omega"

mkdir -p $skewsym_dir
mkdir -p $stdp1_dir
mkdir -p $stdp2_dir
mkdir -p $stdp3_dir


srun -N 1 -n 1 -c $cores -o "$skewsym_dir".out --open-mode=append ./main_wrapper.sh --M $M --spiking --load --use-time-variables --directory $skewsym_dir --omega $omega --step $step --spike-method normal --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
srun -N 1 -n 1 -c $cores -o "$stdp1_dir".out --open-mode=append ./main_wrapper.sh --M $M --spiking --load --use-time-variables --directory $stdp1_dir --omega $omega --step $step --spike-method normal --tau-dynamic $tau_dynamic --tau-trace 0.2 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
srun -N 1 -n 1 -c $cores -o "$stdp2_dir".out --open-mode=append ./main_wrapper.sh --M $M --spiking --load --use-time-variables --directory $stdp2_dir --omega $omega --step $step --spike-method normal --tau-dynamic $tau_dynamic --tau-trace 0.5 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
srun -N 1 -n 1 -c $cores -o "$stdp3_dir".out --open-mode=append ./main_wrapper.sh --M $M --spiking --load --use-time-variables --directory $stdp3_dir --omega $omega --step $step --spike-method normal --tau-dynamic $tau_dynamic --tau-trace 1.0 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
done
done

# Binomial
# Accumulator neuron experiments
beta=0.2
T1=8
T2=3
hidden_size=256
step=0.02 # Keep fixed
batch_size=20
tau_dynamic=0.2
for M in {7,}
do
for omega in {1,7,1e24}
do
skewsym_dir=binom_skewsym_M_"$M"_omega_"$omega"
stdp1_dir=binom_stdp1_M_"$M"_omega_"$omega"
stdp2_dir=binom_stdp2_M_"$M"_omega_"$omega"
stdp3_dir=binom_stdp3_M_"$M"_omega_"$omega"

mkdir -p $skewsym_dir
mkdir -p $stdp1_dir
mkdir -p $stdp2_dir
mkdir -p $stdp3_dir


srun -N 1 -n 1 -c $cores -o "$skewsym_dir".out --open-mode=append ./main_wrapper.sh --M $M --spiking --load --use-time-variables --directory $skewsym_dir --omega $omega --step $step --spike-method binom --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
srun -N 1 -n 1 -c $cores -o "$stdp1_dir".out --open-mode=append ./main_wrapper.sh --M $M --spiking --load --use-time-variables --directory $stdp1_dir --omega $omega --step $step --spike-method binom --tau-dynamic $tau_dynamic --tau-trace 0.2 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
srun -N 1 -n 1 -c $cores -o "$stdp2_dir".out --open-mode=append ./main_wrapper.sh --M $M --spiking --load --use-time-variables --directory $stdp2_dir --omega $omega --step $step --spike-method binom --tau-dynamic $tau_dynamic --tau-trace 0.5 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
srun -N 1 -n 1 -c $cores -o "$stdp3_dir".out --open-mode=append ./main_wrapper.sh --M $M --spiking --load --use-time-variables --directory $stdp3_dir --omega $omega --step $step --spike-method binom --tau-dynamic $tau_dynamic --tau-trace 1.0 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
done
done
