#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 4
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

# # Spiking and nonspiking cepalt for various levels of discretization
# #######################################################################################################################################################
# i=0
# for Kmax in {3,6,9,12}
# 	do
# 		T=$((3*Kmax))
# 		beta=0.2
# 		update_rule="cepalt"
# 		spiking_dir=cepalt_spiking_"$i"
# 		nonspiking_dir=cepalt_nonspiking_"$i"
# 		mkdir -p $spiking_dir
# 		mkdir -p $nonspiking_dir
# 		srun -N 1 -n 1 -c 6 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
# 		srun -N 1 -n 1 -c 6 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
# 		i=$((i+1))
# 	done
#
# for Kmax in {15,18,21,24}
# 	do
# 		T=$((3*Kmax))
# 		beta=0.2
# 		update_rule="cepalt"
# 		spiking_dir=cepalt_spiking_"$i"
# 		nonspiking_dir=cepalt_nonspiking_"$i"
# 		mkdir -p $spiking_dir
# 		mkdir -p $nonspiking_dir
# 		srun -N 1 -n 1 -c 9 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
# 		srun -N 1 -n 1 -c 9 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
# 		i=$((i+1))
# 	done
#
# for Kmax in {27,30}
# 	do
# 		T=$((3*Kmax))
# 		beta=0.2
# 		update_rule="cepalt"
# 		spiking_dir=cepalt_spiking_"$i"
# 		nonspiking_dir=cepalt_nonspiking_"$i"
# 		mkdir -p $spiking_dir
# 		mkdir -p $nonspiking_dir
# 		srun -N 1 -n 1 -c 12 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
# 		srun -N 1 -n 1 -c 12 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $nonspiking_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
# 		i=$((i+1))
# 	done
# ######################################################################################################################################################

# # Plot comparing spiking/nonspiking and different learning rule types
# ########################################################################################################################################################
# i=0
# for batch_size in {25,50,100,200}
#  	do
#  		beta=1.0
# 		T=40
# 		Kmax=15
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
# 		srun -N 1 -n 1 -c 4 -o "$skewsym_spiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $skewsym_spiking_dir --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# 		srun -N 1 -n 1 -c 4 -o "$spiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_slow_dir --spiking --action train --batch-size $batch_size --tau-trace 2.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T  --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 		srun -N 1 -n 1 -c 4 -o "$spiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_med_dir --spiking --action train --batch-size $batch_size --tau-trace 1.44  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 		srun -N 1 -n 1 -c 4 -o "$spiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --load --directory $spiking_stdp_fast_dir --spiking --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 		srun -N 1 -n 1 -c 4 -o "$skewsym_nonspiking_dir".out --open-mode=append ./main_wrapper.sh --load --directory $skewsym_nonspiking_dir --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym &
# 		srun -N 1 -n 1 -c 4 -o "$nonspiking_stdp_slow_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_stdp_slow_dir --action train --batch-size $batch_size --tau-trace 2.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T  --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 		srun -N 1 -n 1 -c 4 -o "$nonspiking_stdp_med_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_stdp_med_dir --action train --batch-size $batch_size --tau-trace 1.44  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp &
# 		srun -N 1 -n 1 -c 4 -o "$nonspiking_stdp_fast_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_stdp_fast_dir --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp &
#   	i=$((i+1))
#  	done
# ########################################################################################################################################################################################


# Spiking and nonspiking cepalt for various levels of discretization (fixed max_fr)
#######################################################################################################################################################
i=0
for Kmax in {3,6,9,12}
	do
		T=$((3*Kmax))
		beta=0.2
		update_rule="cepalt"
		spiking_dir=cepalt_spiking_"$i"
		nonspiking_dir=cepalt_nonspiking_"$i"
		mkdir -p $spiking_dir
		mkdir -p $nonspiking_dir
		srun -N 1 -n 1 -c 6 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_dir --max-fr 5 --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
		srun -N 1 -n 1 -c 6 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_dir --max-fr 5 --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
		i=$((i+1))
	done

for Kmax in {15,18,21,24}
	do
		T=$((3*Kmax))
		beta=0.2
		update_rule="cepalt"
		spiking_dir=cepalt_spiking_"$i"
		nonspiking_dir=cepalt_nonspiking_"$i"
		mkdir -p $spiking_dir
		mkdir -p $nonspiking_dir
		srun -N 1 -n 1 -c 9 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_dir --max-fr 5 --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
		srun -N 1 -n 1 -c 9 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_dir --max-fr 5 --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
		i=$((i+1))
	done

for Kmax in {27,30}
	do
		T=$((3*Kmax))
		beta=0.2
		update_rule="cepalt"
		spiking_dir=cepalt_spiking_"$i"
		nonspiking_dir=cepalt_nonspiking_"$i"
		mkdir -p $spiking_dir
		mkdir -p $nonspiking_dir
		srun -N 1 -n 1 -c 12 -o "$spiking_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_dir --max-fr 5 --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule &
		srun -N 1 -n 1 -c 12 -o "$nonspiking_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_dir --max-fr 5 --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule  &
		i=$((i+1))
	done
######################################################################################################################################################


# Compare spiking/nonspiking cepalt/skewsym
######################################################################################################################################################
T=40
Kmax=15
beta=0.5
step=1.0
max_fr=1.0
spiking_skewsym_dir=spiking_skewsym
nonspiking_skewsym_dir=nonspiking_skewsym
spiking_cepalt_dir=spiking_cepalt
nonspiking_cepalt_dir=nonspiking_cepalt

mkdir -p $spiking_skewsym_dir
mkdir -p $nonspiking_skewsym_dir
mkdir -p $spiking_cepalt_dir
mkdir -p $nonspiking_cepalt_dir

srun -N 1 -n 1 -c 4 -o "$spiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_skewsym_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 50 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym &
srun -N 1 -n 1 -c 4 -o "$nonspiking_skewsym_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_skewsym_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 50 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym &
srun -N 1 -n 1 -c 4 -o "$spiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --directory $spiking_cepalt_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 50 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule cepalt &
srun -N 1 -n 1 -c 4 -o "$nonspiking_cepalt_dir".out --open-mode=append ./main_wrapper.sh --directory $nonspiking_cepalt_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 50 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule cepalt &
######################################################################################################################################################


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
