#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 4
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

# Plot 1 (Variable N, compensating dt)
i=0
for Kmax in {3,6,9,12}
	do
		T=$((3*Kmax))
		beta=0.2
		update_rule="cepalt"
		spiking_dir=cepalt_spiking_b_"$i"
		nonspiking_dir=cepalt_nonspiking_b_"$i"
		mkdir -p $spiking_dir
		mkdir -p $nonspiking_dir
		srun -N 1 -n 1 -c 6 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
		srun -N 1 -n 1 -c 6 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
		# srun -N 1 -n 1 -c 6 -o "$spiking_dir".out --open-mode=append python -u main.py --load True --directory $spiking_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
		# srun -N 1 -n 1 -c 6 -o "$nonspiking_dir".out --open-mode=append python -u main.py --load True --directory $nonspiking_directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
		i=$((i+1))
	done

for Kmax in {15,18,21,24}
	do
		T=$((3*Kmax))
		beta=0.2
		update_rule="cepalt"
		spiking_dir=cepalt_spiking_b_"$i"
		nonspiking_dir=cepalt_nonspiking_b_"$i"
		mkdir -p $spiking_dir
		mkdir -p $nonspiking_dir
		srun -N 1 -n 1 -c 9 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
		srun -N 1 -n 1 -c 9 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
		# srun -N 1 -n 1 -c 9 -o "$spiking_dir".out --open-mode=append python -u main.py --load True --directory $spiking_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
		# srun -N 1 -n 1 -c 9 -o "$nonspiking_dir".out --open-mode=append python -u main.py --load True --directory $nonspiking_directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
		i=$((i+1))
	done

for Kmax in {27,30}
	do
		T=$((3*Kmax))
		beta=0.2
		update_rule="cepalt"
		spiking_dir=cepalt_spiking_b_"$i"
		nonspiking_dir=cepalt_nonspiking_b_"$i"
		mkdir -p $spiking_dir
		mkdir -p $nonspiking_dir
		srun -N 1 -n 1 -c 12 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
		srun -N 1 -n 1 -c 12 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
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

# beta=0.2
# dt=0.5
# update_rule="skewsym"
# directory='skewsym_dt=05'
# mkdir $directory
# nohup python -u main.py --directory $directory --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule $update_rule >> '$directory'/log.out &
#
# beta=0.2
# dt=0.2
# update_rule="skewsym"
# directory='skewsym_dt=02'
# mkdir $directory
# nohup python -u main.py --directory $directory --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule $update_rule >> '$directory'/log.out


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
#        nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule $update_rule >> fast_"$update_rule"_"$i".out &
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
#         nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 80 --Kmax 30 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule  skewsym >> slow_"$update_rule"_"$i".out &
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
#     nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.005 --epochs 30 --T 50 --Kmax 25 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule stdp >> stdp_"$i".out &
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
#     nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta $beta --dt $dt --learning-rule 'vf' --update-rule 'asym2' --cep >> asym_"$i".out &
#     i=$((i+1))
#   done
# done
#
# # Kmax
# for kmax in {15,20,25,30,35,40}
#   do
#     nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax $kmax --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  stdp > stdp_kmax_"$kmax".out &
#   done
#
# # Kmax
# for kmax in {15,20,25,30,35,40}
#   do
#   nohup python -u main.py --action train --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 80 --Kmax $kmax --beta 0.2 --dt 0.2 --cep > cont_kmax_"$kmax".out &
#   done
