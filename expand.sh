
#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 11
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

epochs=100
cores=20

# Binomial
# Accumulator neuron experiments
beta=0.2
T1=8
T2=3
# hidden_size=256
# step=0.2 #0.02 # Keep fixed
batch_size=20
tau_dynamic=0.2
omega=1
max_fr=6
# for M in {8,} # 1,4,8,16,32
# do
# for omega in {1,8,4096} # 1,8,32,4096
# do
# for max_fr in {8,} # 4,8,16
# do

for up_sample in {1,4,16}
do
for hidden_size in {256,384,512}
do

# skewsym_dir=skewsym_M_"$M"_omega_"$omega"_freq_"$max_fr"
# fast_stdp_dir=fast_stdp_M_"$M"_omega_"$omega"_freq_"$max_fr"
# slow_stdp_dir=slow_stdp_M_"$M"_omega_"$omega"_freq_"$max_fr"

skewsym_dir=skewsym_up_"$up_sample"_hid_"$hidden_size"
fast_stdp_dir=fast_stdp_up_"$up_sample"_hid_"$hidden_size"
slow_stdp_dir=slow_stdp_up_"$up_sample"_hid_"$hidden_size"

mkdir -p $skewsym_dir
mkdir -p $fast_stdp_dir
mkdir -p $slow_stdp_dir

# srun -N 1 -n 1 -c $((M*2)) -o "$skewsym_dir".out --open-mode=append ./main_wrapper.sh --M $M --spiking --load --use-time-variables --directory $skewsym_dir --omega $omega --max-fr $max_fr --spike-method binomial --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# srun -N 1 -n 1 -c $((M*2)) -o "$fast_stdp_dir".out --open-mode=append ./main_wrapper.sh --M $M --spiking --load --use-time-variables --directory $fast_stdp_dir --omega $omega --max-fr $max_fr --spike-method binomial --tau-dynamic $tau_dynamic --tau-trace 0.2 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &
# srun -N 1 -n 1 -c $((M*2)) -o "$slow_stdp_dir".out --open-mode=append ./main_wrapper.sh --M $M --spiking --load --use-time-variables --directory $slow_stdp_dir --omega $omega --max-fr $max_fr --spike-method binomial --tau-dynamic $tau_dynamic --tau-trace 0.5 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size 784 --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --learning-rule stdp --update-rule stdp &

srun -N 1 -n 1 -c $cores -o "$skewsym_dir".out --open-mode=append ./main_wrapper.sh --up-sample $up_sample --spiking --load --use-time-variables --directory $skewsym_dir --omega $omega --max-fr $max_fr --spike-method binomial --tau-dynamic $tau_dynamic --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size $((784*up_sample)) --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --update-rule skewsym &
srun -N 1 -n 1 -c $cores -o "$fast_stdp_dir".out --open-mode=append ./main_wrapper.sh --up-sample $up_sample --spiking --load --use-time-variables --directory $fast_stdp_dir --omega $omega --max-fr $max_fr --spike-method binomial --tau-dynamic $tau_dynamic --tau-trace 0.2 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size $((784*up_sample)) --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --update-rule stdp &
srun -N 1 -n 1 -c $cores -o "$slow_stdp_dir".out --open-mode=append ./main_wrapper.sh --up-sample $up_sample --spiking --load --use-time-variables --directory $slow_stdp_dir --omega $omega --max-fr $max_fr --spike-method binomial --tau-dynamic $tau_dynamic --tau-trace 0.5 --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 $hidden_size $((784*up_sample)) --lr_tab 0.0028 0.0056 --epochs $epochs --T1 $T1 --T2 $T2 --beta $beta --cep --update-rule stdp &

done
done
