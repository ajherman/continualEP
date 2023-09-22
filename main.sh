#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH--time 10:00:00
#SBATCH -N 4
#SBATCH -p shared-gpu
module load miniconda3
source activate /vast/home/ajherman/miniconda3/envs/pytorch
# First plot (Fixed N, variable dt)
# i=0
# for dt in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
# 	do
# 		beta=0.2
# 		update_rule="cepalt"
# 		spiking_directory=cepalt_spiking_"$i"
# 		nonspiking_directory=cepalt_nonspiking_"$i"
# 		mkdir -p $spiking_directory
# 		mkdir -p $nonspiking_directory
# 		nohup python -u main.py --directory $spiking_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --dt $dt --cep --learning-rule stdp --update-rule $update_rule >> spiking_log_"$i".out &
# 		nohup python -u main.py --directory $nonspiking_directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --dt $dt --cep --learning-rule stdp --update-rule $update_rule >> nonspiking_log_"$i".out &
# 		i=$((i+1))
# 	done

# Second plot (Variable N, compensating dt)
i=0
for Kmax in {3,6,9,12}
	do
		T=$((3*Kmax))
		beta=0.2
		update_rule="cepalt"
		spiking_directory=cepalt_spiking_b_"$i"
		nonspiking_directory=cepalt_nonspiking_b_"$i"
		mkdir -p $spiking_directory
		mkdir -p $nonspiking_directory
		srun -N 1 -n 1 -c 6 -o spiking_log_b_"$i".out --open-mode=append python -u main.py --load True --directory $spiking_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
		srun -N 1 -n 1 -c 6 -o nonspiking_log_b_"$i".out --open-mode=append python -u main.py --load True --directory $nonspiking_directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
		i=$((i+1))
	done

for Kmax in {15,18,21,24}
	do
		T=$((3*Kmax))
		beta=0.2
		update_rule="cepalt"
		spiking_directory=cepalt_spiking_b_"$i"
		nonspiking_directory=cepalt_nonspiking_b_"$i"
		mkdir -p $spiking_directory
		mkdir -p $nonspiking_directory
		srun -N 1 -n 1 -c 9 -o spiking_log_b_"$i".out --open-mode=append python -u main.py --load True --directory $spiking_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
		srun -N 1 -n 1 -c 9 -o nonspiking_log_b_"$i".out --open-mode=append python -u main.py --load True --directory $nonspiking_directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
		i=$((i+1))
	done

for Kmax in {27,30}
	do
		T=$((3*Kmax))
		beta=0.2
		update_rule="cepalt"
		spiking_directory=cepalt_spiking_b_"$i"
		nonspiking_directory=cepalt_nonspiking_b_"$i"
		mkdir -p $spiking_directory
		mkdir -p $nonspiking_directory
		srun -N 1 -n 1 -c 12 -o spiking_log_b_"$i".out --open-mode=append python -u main.py --load True --directory $spiking_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
		srun -N 1 -n 1 -c 12 -o nonspiking_log_b_"$i".out --open-mode=append python -u main.py --load True --directory $nonspiking_directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
		i=$((i+1))
	done

# i=0
# for Kmax in {5,10,15,20,25,30,35,40}
# 	do
# 		T=$((3*Kmax))
# 		beta=0.2
# 		spiking_directory=skewsym_spiking_"$i"
# 		nonspiking_directory=skewsym_nonspiking_"$i"
# 		stdp_directory=stdp_"$i"
# 		mkdir -p $spiking_directory
# 		mkdir -p $nonspiking_directory
# 		mkdir -p $stdp_directory
# 		nohup python -u main.py --directory $spiking_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym >> skewsym_spiking_log_"$i".out &
# 		nohup python -u main.py --directory $stdp_directory --spiking True --action train --trace-decay 0.5  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp >> stdp_log_"$i".out &
# 		# nohup python -u main.py --directory $stdp_directory --spiking True --action train --trace-decay 0.38197  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp >> stdp_log_"$i".out &
# 		nohup python -u main.py --directory $nonspiking_directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym >> skewsym_nonspiking_log_"$i".out &
# 		i=$((i+1))
# 	done

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
	# 		nohup python -u main.py --directory $spiking_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --dt $dt --cep --learning-rule stdp --update-rule skewsym >> skewsym_spiking_log_"$i".out &
	# 		nohup python -u main.py --directory $stdp_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --dt $dt --cep --learning-rule stdp --update-rule stdp >> stdp_log_"$i".out &
	# 		nohup python -u main.py --directory $nonspiking_directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --dt $dt --cep --learning-rule stdp --update-rule skewsym >> skewsym_nonspiking_log_"$i".out &
	# 		i=$((i+1))
	# 	done

# i=0
# for dt in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
# 	do
# 	beta=0.2
# 	update_rule="cepalt"
# 	directory=cepalt_"$i"
# 	mkdir -p $directory
# 	nohup python -u main.py --directory $directory --load True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --dt $dt --cep --learning-rule stdp --update-rule cepalt >> log_"$i".out &
# 	i=$((i+1))
# 	done
