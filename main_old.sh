#!/bin/bash

# Discrete
#i=0
#for beta in {0.1,0.2,0.5,0.8,1.0}
#   do
#     echo beta = $beta >> disc_"$i".out
#     echo "" >> disc_"$i".out
#     nohup python -u main.py --action train --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta $beta --cep --update-rule skew-sym1 >> disc_"$i".out &
#     i=$((i+1))
#   done
# # Fast
# for update_rule in {cepalt}
#   do
#     i=0
#     for beta in {0.2,0.5,1.0}
#     do
#       for dt in {0.2,0.5,1.0}
#       do
#         echo rule = $update_rule , beta = $beta , dt = $dt >> nonspike_fast_"$i".out
#         echo "" >> nonspike_fast_"$i".out
#         nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule $update_rule >> fast_"$update_rule"_"$i".out &
#         nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule $update_rule >> fast_"$update_rule"_"$i".out &
#         i=$((i+1))
#       done
#     done
#   done

#beta=0.2
#dt=0.1
#update_rule="cepalt"
#directory='cepalt_dt=03'
#mkdir -p $directory
#nohup python -u main.py --directory $directory --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta $beta --dt $dt --cep --learning-rule stdp --update-rule $update_rule >> log_03.out &

# Compare spiking and nonspiking (Variable N, compensating step)
i=0
for Kmax in {3,6,9,12,15,18}
	do
		T=$((3*Kmax))
		update_rule="cepalt"
		spiking_dir=cepalt_spiking_"$i"
		nonspiking_dir=cepalt_nonspiking_"$i"
		mkdir -p $spiking_dir
		mkdir -p $nonspiking_dir
		nohup python -u main.py --directory $spiking_dir --spiking --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule >> "$spiking_dir".out &
		nohup python -u main.py --directory $nonspiking_dir --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta 0.2 --cep --learning-rule stdp --update-rule $update_rule >> "$nonspiking_dir".out &
		i=$((i+1))
	done

# Plot comparing three types
i=0
for batch_size in {200,100,20,40} # 10,20,40
 	do
 		beta=0.9
		T=40
		Kmax=15
 		spiking_dir=skewsym_spiking_"$i"
 		nonspiking_dir=skewsym_nonspiking_"$i"
 		stdp_slow_dir=stdp_slow_"$i"
		stdp_med_dir=stdp_med_"$i"
		stdp_fast_dir=stdp_fast_"$i"
 		mkdir -p $spiking_dir
 		mkdir -p $nonspiking_dir
 		mkdir -p $stdp_slow_dir
		mkdir -p $stdp_med_dir
		mkdir -p $stdp_fast_dir
 		nohup python -u main.py --load --directory $spiking_dir --spiking --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym >> "$spiking_dir".out &
 		nohup python -u main.py --load --directory $nonspiking_dir --action train --batch-size $batch_size --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym >> "$nonspiking_dir".out &
		nohup python -u main.py --load --directory $stdp_slow_dir --spiking --action train --batch-size $batch_size --tau-trace 2.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T  --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp >> "$stdp_slow_dir".out &
		nohup python -u main.py --load --directory $stdp_med_dir --spiking --action train --batch-size $batch_size --tau-trace 1.44  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp >> "$stdp_med_dir".out &
		nohup python -u main.py --load --directory $stdp_fast_dir --spiking --action train --batch-size $batch_size --tau-trace 1.0  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --step 1.0 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp >> "$stdp_fast_dir".out &
 		i=$((i+1))
 	done

# # Plot comparing different decay rates
# i=1
# for beta in {0.5,}
# 	do
# 	 		spiking_directory=skewsym_spiking_b_"$i"
# 	 		nonspiking_directory=skewsym_nonspiking_b_"$i"
# 			stdp_0_directory=stdp_0_"$i"
# 			stdp_1_directory=stdp_1_"$i"
# 			stdp_2_directory=stdp_2_"$i"
# 			stdp_3_directory=stdp_3_"$i"
# 	 		mkdir -p $spiking_directory
# 	 		mkdir -p $nonspiking_directory
# 	 		mkdir -p $stdp_0_directory
# 	 		mkdir -p $stdp_1_directory
# 	 		mkdir -p $stdp_2_directory
# 	 		mkdir -p $stdp_3_directory
# 			nohup python -u main.py --directory $spiking_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 20 --T 40 --Kmax 15 --beta $beta --cep --learning-rule stdp --update-rule skewsym >> skewsym_spiking_log_b_"$i".out &
# 			nohup python -u main.py --directory $nonspiking_directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 20 --T 40 --Kmax 15 --beta $beta --cep --learning-rule stdp --update-rule skewsym >> skewsym_nonspiking_log_b_"$i".out &
# 			nohup python -u main.py --directory $stdp_0_directory --spiking True --action train --tau-trace 2.847366  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 20 --T 40 --Kmax 15 --beta $beta --cep --learning-rule stdp --update-rule stdp >> stdp_log_0_"$i".out &
# 			nohup python -u main.py --directory $stdp_1_directory --spiking True --action train --tau-trace 0.8411019  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 20 --T 40 --Kmax 15 --beta $beta --cep --learning-rule stdp --update-rule stdp >> stdp_log_1_"$i".out &
# 			nohup python -u main.py --directory $stdp_2_directory --spiking True --action train --tau-trace 0.4328085  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 20 --T 40 --Kmax 15 --beta $beta --cep --learning-rule stdp --update-rule stdp >> stdp_log_2_"$i".out &
# 	 		nohup python -u main.py --directory $stdp_3_directory --spiking True --action train --tau-trace 0.327407  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 20 --T 40 --Kmax 15 --beta $beta --cep --learning-rule stdp --update-rule stdp >> stdp_log_3_"$i".out &
# 		done

# # Plot comparing skewsym and cepalt
# j=0
# for beta in {1.0,0.5,0.2}
# 	do
# 	for Kmax in {5,10,15}
# 	do
# 		T=$((3*Kmax))
# 		cepalt_dir=cepalt_"$j"
# 		skewsym_dir=skewsym_"$j"
# 		mkdir -p $cepalt_dir
# 		mkdir -p $skewsym_dir
# 		nohup python -u main.py --directory $cepalt_dir --load True --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 20 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym >> cepalt_"$j".out &
# 		nohup python -u main.py --directory $skewsym_dir --load True --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 20 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym >> skewsym_"$j".out &
# 		j=$((j+1))
# 	done
# done

# # Test discretization scheme
# j=0
# for Kmax in {3,6,9,12,15}
# do
# 	T=$((3*Kmax))
# 	directory=disc_test_"$j"
# 	mkdir -p $directory
# 	nohup python -u main.py --directory $directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 25 --T $T --Kmax $Kmax --beta 0.5 --cep --learning-rule stdp --update-rule skewsym >> "$directory".out &
# 	j=$((j+1))
# done


# i=0
# for Kmax in {5,10,15}
#  	do
# 		for beta in {1.0,0.5,0.2}
# 		do
# 	 		T=$((3*Kmax))
# 	 		spiking_directory=skewsym_spiking_"$i"
# 	 		nonspiking_directory=skewsym_nonspiking_"$i"
# 	 		stdp_directory=stdp_"$i"
# 	 		mkdir -p $spiking_directory
# 	 		mkdir -p $nonspiking_directory
# 	 		mkdir -p $stdp_directory
# 	 		nohup python -u main.py --directory $spiking_directory --spiking True --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym >> skewsym_spiking_log_"$i".out &
# 	 		nohup python -u main.py --directory $stdp_directory --spiking True --action train --trace-decay 0.5  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp >> stdp_log_"$i".out &
# 	 		# nohup python -u main.py --directory $stdp_directory --spiking True --action train --trace-decay 0.38197  --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule stdp >> stdp_log_"$i".out &
# 	 		nohup python -u main.py --directory $nonspiking_directory --spiking False --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T $T --Kmax $Kmax --beta $beta --cep --learning-rule stdp --update-rule skewsym >> skewsym_nonspiking_log_"$i".out &
# 	 		i=$((i+1))
# 		done
#  	done

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
