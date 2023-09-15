 # # EP
# nohup python -u main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.04 0.08 --epochs 30 --T 30 --Kmax 10 --beta 0.1
#
# # EP
# nohup python -u main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.005 0.05 0.2 --epochs 50 --T 100 --Kmax 20 --beta 0.5

# # CEP
#nohup python -u main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep > results1.out &
#
# # CEP
# nohup python -u main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep >> no_bias1.out &

# CEP
# nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep --update_rule cep > old_rule.out &

# Experiment with new rule
#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep  --update_rule cep-alt > new_rule.out &

# CEP
#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 256 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep > no_bias3.out &


# Experiments! 1 layer - compare different symmetric rules

# nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep --update_rule cep > old_rule.out &

# Experiment comparing different symmetric rules

#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep --update-rule cep > cep.out &
#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep --update-rule cep-alt > cepalt.out &
#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep --update-rule skew-sym1 > skewsym.out &
#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep --update-rule skew-sym2 > skewsym2.out &
#nohup python -u main.py --action train --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --dt 0.5 --cep --update-rule skew-sym1 > skewsym3.out &
#nohup python -u main.py --action train --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 120 --Kmax 45 --beta 0.2 --dt 0.2 --cep --update-rule skew-sym1 > skewsym4.out &
#nohup python -u main.py --action train --activation-function hardsigm  --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --dt 0.5 --cep --update-rule skew-sym1 > skewsym5.out

#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --dt 0.5 --cep --learning-rule stdp --update-rule  skew1 > spike1.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --dt 1.0 --cep --learning-rule stdp --update-rule  skew1 > spike2.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.5 --dt 0.5 --cep --learning-rule stdp --update-rule  skew1 > spike3.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.5 --dt 1.0 --cep --learning-rule stdp --update-rule  skew1 > spike4.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.8 --dt 0.5 --cep --learning-rule stdp --update-rule  skewsym > spike5.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.8 --dt 1.0 --cep --learning-rule stdp --update-rule  skewsym > spike6.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  skewsym > spike7.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 1.0 --dt 1.0 --cep --learning-rule stdp --update-rule  skewsym > spike8.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 45 --Kmax 15 --beta 0.1 --dt 0.5 --cep --learning-rule stdp --update-rule  skewsym > spike7.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 45 --Kmax 15 --beta 0.2 --dt 0.5 --cep --learning-rule stdp --update-rule  skewsym > spike8.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 45 --Kmax 15 --beta 0.5 --dt 0.5 --cep --learning-rule stdp --update-rule  skewsym > spike9.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 45 --Kmax 15 --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  skewsym > spike10.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 45 --Kmax 15 --beta 0.1 --dt 0.2 --cep --learning-rule stdp --update-rule  skewsym > spike11.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 45 --Kmax 15 --beta 0.2 --dt 0.2 --cep --learning-rule stdp --update-rule  skewsym > spike12.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 45 --Kmax 15 --beta 0.5 --dt 0.2 --cep --learning-rule stdp --update-rule  skewsym > spike13.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 45 --Kmax 15 --beta 1.0 --dt 0.2 --cep --learning-rule stdp --update-rule  skewsym > spike14.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 45 --Kmax 15 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  skewsym > spike15.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 60 --Kmax 20 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  skewsym > spike16.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 90 --Kmax 30 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  skewsym > spike17.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 120 --Kmax 40 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  skewsym > spike18.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 60 --Kmax 20 --beta 1.0 --dt 0.05 --cep --learning-rule stdp --update-rule  skewsym > spike19.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 90 --Kmax 30 --beta 1.0 --dt 0.05 --cep --learning-rule stdp --update-rule  skewsym > spike20.out & 
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 120 --Kmax 40 --beta 1.0 --dt 0.05 --cep --learning-rule stdp --update-rule  skewsym > spike21.out &

#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 120 --Kmax 40 --beta 1.0 --dt 0.05 --cep --learning-rule stdp --update-rule  skewsym > spike22.out &

#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 120 --Kmax 40 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  skewsym > stdp.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 120 --Kmax 40 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  skewsym > stdp.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 120 --Kmax 40 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule stdp > stdp.out &

# nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 50 --Kmax 25 --beta 1.0 --dt 0.05 --cep --learning-rule stdp --update-rule  stdp > stdp1.out &
# nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.005 --epochs 30 --T 50 --Kmax 25 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp2.out &
# nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.005 --epochs 30 --T 50 --Kmax 25 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp3.out &



#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.0025 --epochs 30 --T 50 --Kmax 25 --beta 1.0 --dt 0.05 --cep --learning-rule stdp --update-rule  stdp > stdp1.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.0025 --epochs 30 --T 50 --Kmax 25 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp2.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.0025 --epochs 30 --T 50 --Kmax 25 --beta 1.0 --dt 0.2 --cep --learning-rule stdp --update-rule  stdp > stdp3.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.005 --epochs 30 --T 50 --Kmax 25 --beta 1.0 --dt 0.05 --cep --learning-rule stdp --update-rule  stdp > stdp4.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.005 --epochs 30 --T 50 --Kmax 25 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp5.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.005 --epochs 30 --T 50 --Kmax 25 --beta 1.0 --dt 0.2 --cep --learning-rule stdp --update-rule  stdp > stdp6.out &

#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.0025 --epochs 30 --T 50 --Kmax 25 --beta 0.5 --dt 0.05 --cep --learning-rule stdp --update-rule  stdp > stdp7.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.0025 --epochs 30 --T 50 --Kmax 25 --beta 0.5 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp8.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.0025 --epochs 30 --T 50 --Kmax 25 --beta 0.5 --dt 0.2 --cep --learning-rule stdp --update-rule  stdp > stdp9.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.005 --epochs 30 --T 50 --Kmax 25 --beta 0.5 --dt 0.05 --cep --learning-rule stdp --update-rule  stdp > stdp10.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.005 --epochs 30 --T 50 --Kmax 25 --beta 0.5 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp11.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0025 0.005 --epochs 30 --T 50 --Kmax 25 --beta 0.5 --dt 0.2 --cep --learning-rule stdp --update-rule  stdp > stdp12.out &



#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax 50 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp13.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0025 0.0025 0.005 --epochs 30 --T 100 --Kmax 50 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp14.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.002 0.002 0.002 --epochs 30 --T 100 --Kmax 50 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp15.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.001 0.001 0.001 --epochs 30 --T 100 --Kmax 50 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp16.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 80 --Kmax 25 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp17.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0025 0.0025 0.05 --epochs 30 --T 80 --Kmax 25 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp18.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.002 0.002 0.002 --epochs 30 --T 80 --Kmax 25 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp19.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.001 0.001 0.001 --epochs 30 --T 80 --Kmax 25 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp20.out &


#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax 20 --beta 1.0 --dt 0.05 --cep --learning-rule stdp --update-rule stdp > stdp21.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax 20 --beta 1.0 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp22.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax 20 --beta 1.0 --dt 0.2 --cep --learning-rule stdp --update-rule  stdp > stdp23.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax 20 --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  stdp > stdp24.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax 20 --beta 0.2 --dt 0.05 --cep --learning-rule stdp --update-rule  stdp > stdp25.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax 20 --beta 0.2 --dt 0.1 --cep --learning-rule stdp --update-rule  stdp > stdp26.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax 20 --beta 0.2 --dt 0.2 --cep --learning-rule stdp --update-rule  stdp > stdp27.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax 20 --beta 0.2 --dt 0.5 --cep --learning-rule stdp --update-rule  stdp > stdp28.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax 20 --beta 0.2 --dt 1.0 --cep --learning-rule stdp --update-rule  stdp > stdp29.out &

#i=0
#for beta in {0.1,0.2,0.5,0.8,1.0}
#do
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta $beta --dt 0.5 --cep --learning-rule stdp --update-rule  skewsym > skewsym_"$i".out &
#i=$((i+1))
#done

j=0
for beta in {0.1,0.2,0.5,0.8,1.0}
do
nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta $beta --dt 1.0  --cep --learning-rule stdp --update-rule skewsym > skewsym_discrete_"$j".out &
j=$((j+1))
done

#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.5 --dt 0.5 --cep --learning-rule stdp --update-rule  skewsym > skewsym2.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.8 --dt 0.5 --cep --learning-rule stdp --update-rule  skewsym > skewsym3.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  skewsym > skewsym4.out &

#for Kmax in {15,25,40}
#do
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax $Kmax --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  stdp > stdp_kmax_"${Kmax}".out &
#done

#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.001 0.01 0.05 --epochs 30 --T 100 --Kmax 20 --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  stdp > stdp31.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 100 --Kmax 40 --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  stdp > stdp32.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.001 0.01 0.05 --epochs 30 --T 100 --Kmax 40 --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  stdp > stdp33.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.0002 0.002 0.01 --epochs 30 --T 50 --Kmax 20 --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  stdp > stdp34.out &
#nohup python -u main.py --action train --activation-function hardsigm --size_tab 10 512 512 784 --lr_tab 0.001 0.01 0.05 --epochs 30 --T 50 --Kmax 20 --beta 1.0 --dt 0.5 --cep --learning-rule stdp --update-rule  stdp > stdp35.out &






# Experiments! 2 layer - check symmetric rule

#nohup python -u main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep --update_rule cep-alt >> new_rule1.out &
#nohup python -u main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep --update_rule cep-alt >> new_rule2.out &
#nohup python -u main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep --update_rule cep-alt >> new_rule3.out &
#nohup python -u main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep --update_rule cep-alt >> new_rule4.out &


#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep --no-reset --update_rule cep-alt > noreset1.out &
#nohup python -u main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep --no-reset --update_rule cep-alt > noreset2.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 40 --Kmax 15 --beta 0.2 --dt 0.5 --cep --no-reset --no-rhop --update-rule cep-alt > test1.out &
#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep --no-reset --no-reset --update_rule cep > noreset4.out &

# Experiments! 1 layer - compare different epsilons
# nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 40 --Kmax 15 --beta 0.2 --dt 1.0 --cep --device-label 0 > cont1.out &
# nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 40 --Kmax 15 --beta 0.2 --dt 0.5 --cep --no-rhop --device-label 0 > cont2.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 40 --Kmax 15 --beta 0.2 --dt 0.2 --cep > test.out

# nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 80 --Kmax 15 --beta 0.2 --dt 0.2 --cep --no-reset > cont4.out &
# nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 80 --Kmax 30 --beta 0.2 --dt 0.2 --cep --no-reset > cont5.out &
# nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 120 --Kmax 15 --beta 0.2 --dt 0.2 --cep --no-reset > cont6.out &
# nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 120 --Kmax 45 --beta 0.2 --dt 0.2 --cep --no-reset > cont7.out &


for kmax in {15,25,40}
do
nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 80 --Kmax $kmax --beta 0.2 --dt 0.2 --cep > cont_kmax_"$kmax".out &
done

#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 80 --Kmax 30 --beta 0.2 --dt 0.2 --cep > cont5.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 120 --Kmax 15 --beta 0.2 --dt 0.2 --cep > cont6.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 120 --Kmax 45 --beta 0.2 --dt 0.2 --cep > cont7.out &

#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep --update_rule cep --device-label 0 > old_rule.out &

#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep --update_rule cep-alt --device-label 0 > new_rule.out &


# Experiments! 1 layer - compare different epsilons
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 40 --Kmax 15 --beta 0.2 --dt 1.0 --cep --device-label 1 > cont1.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 40 --Kmax 15 --beta 0.2 --dt 0.5 --cep --device-label 2 > cont2.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 40 --Kmax 15 --beta 0.2 --dt 0.2 --cep --device-label 3 > cont3.out &

# Experiments! 2 layer
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epoch 150 --T 100 --Kmax 20 --beta 0.5 --dt 1.0 --no-rhop --plain-data --cep --update-rule cep > cont.out &
#nohup python -u main.py --action 'train' --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epoch 150 --T 100 --Kmax 20 --beta 0.5 --discrete --cep --update-rule cep > disc.out &


#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epoch 150 --T 100 --Kmax 20 --beta 0.5 --dt 0.5 --cep --update-rule cep > exp000.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epoch 150 --T 100 --Kmax 20 --beta 0.5 --dt 0.5 --cep --update-rule cep-alt > exp010.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epoch 150 --T 100 --Kmax 20 --beta 0.5 --dt 0.5 --cep --no-reset --update-rule cep > exp100.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epoch 150 --T 100 --Kmax 20 --beta 0.5 --dt 0.5 --cep --no-reset --update-rule cep-alt > exp110.out &


#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epoch 150 --T 100 --Kmax 20 --beta 0.5 --dt 0.5 --cep --no-rhop --update-rule cep > exp001.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epoch 150 --T 100 --Kmax 20 --beta 0.5 --dt 0.5 --cep --no-rhop --update-rule cep-alt > exp011.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epoch 150 --T 100 --Kmax 20 --beta 0.5 --dt 0.5 --cep --no-rhop --no-reset --update-rule cep > exp101.out &
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epoch 150 --T 100 --Kmax 20 --beta 0.5 --dt 0.5 --cep --no-rhop --no-reset --update-rule cep-alt > exp111.out &

# Skew and asym experiments
#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --learning-rule 'vf' --update-rule 'asym1' --cep > asym1disc.out &
#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --learning-rule 'vf' --update-rule 'asym2' --cep > asym2disc.out &
#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --learning-rule 'vf' --update-rule 'skew1' --cep > skew1disc.out &
#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --learning-rule 'vf' --update-rule 'skew2' --cep > skew2disc.out &

#nohup python -u main.py --action 'train' --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --learning-rule 'vf' --update-rule 'asym1' --cep > asym1cont.out &
#nohup python -u main.py --action 'train' --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --learning-rule 'vf' --update-rule 'asym2' --cep > asym2cont.out &
#nohup python -u main.py --action 'train' --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --learning-rule 'vf' --update-rule 'skew1' --cep > skew1cont.out &
#nohup python -u main.py --action 'train' --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 30 --T 40 --Kmax 15 --beta 0.2 --learning-rule 'vf' --update-rule 'skew2' --cep > skew2cont.out &


# CVF
#nohup python -u main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.0038 0.0076 --epochs 100 --T 40 --Kmax 15 --beta 0.20 --cep --learning-rule 'vf' --update-rule 'asym1' > test2.out &


# CVF
#nohup python -u main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00016 0.0016 0.009 --epochs 150 --T 100 --Kmax 20 --beta 0.35 > results6.out & --cep --learning-rule 'vf' --randbeta 0.5

# # Continuous EP
# nohup python -u main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.005 --T 800 --Kmax 80 --seed 0
#
# # Continuous CEP
# nohup python -u main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.005 --T 800 --Kmax 80 --cep --lr_tab 0.00002 0.00002
#
# # Continuous CVF
# nohup python -u main.py --action 'plotcurves' --no-clamp --batch-size 1 --size_tab 10 512 784 --activation-function 'tanh' --dt 0.08 --beta 0.005 --T 800 --Kmax 80 --learning-rule 'vf' --lr_tab 0.00002 0.00002 --cep
