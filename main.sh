# # EP
# nohup python -u main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.04 0.08 --epochs 30 --T 30 --Kmax 10 --beta 0.1
#
# # EP
# nohup python -u main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.005 0.05 0.2 --epochs 50 --T 100 --Kmax 20 --beta 0.5

# # CEP
# nohup python -u main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep >> results1.out &
#
# # CEP
# nohup python -u main.py --action 'train' --discrete --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep >> no_bias1.out &

# CEP
nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep > old_rule.out &

# Experiment with new rule
nohup python -u main.py --action 'train' --discrete --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 100 --T 40 --Kmax 15 --beta 0.2 --cep  --use-alt-update > new_rule.out &

# CEP
#nohup python -u main.py --action 'train' --discrete --size_tab 10 256 256 784 --lr_tab 0.00018 0.0018 0.01 --epochs 150 --T 100 --Kmax 20 --beta 0.5 --cep > no_bias3.out &


# Experiments!
#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 256 784 --lr_tab 0.0028 0.0056 --epochs 15 --T 40 --Kmax 15 --beta 0.2 --dt 1.0 --cep --device-label 3 > exp1.out &

#nohup python -u main.py --action 'train' --no-clamp --size_tab 10 512 512 784 --lr_tab 0.00018 0.0018 0.01 --epoch 150 --T 100 --Kmax 20 --beta 0.5 --dt 1.0 --cep --device-label 3 > exp2.out

# CVF
#nohup python -u main.py --action 'train' --discrete --size_tab 10 512 784 --lr_tab 0.0038 0.0076 --epochs 100 --T 40 --Kmax 15 --beta 0.20 --cep --learning-rule 'vf' > results5.out & --randbeta 0.5


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
