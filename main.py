import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
import datetime
import random
from netClasses import *
from netFunctions import *
from plotFunctions import *
import os

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed()


#***************ICLR VERSION***************#

parser = argparse.ArgumentParser(description='Equilibrium Propagation with Continual Weight Updates')

# Optimization arguments
parser.add_argument(
    '--batch-size',
    type=int,
    default=20,
    metavar='N',
    help='input batch size for training (default: 20)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    metavar='N',
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=1,
    metavar='N',
help='number of epochs to train (default: 1)')
parser.add_argument(
    '--lr_tab',
    nargs = '+',
    type=float,
    default=[0.05, 0.1],
    metavar='LR',
    help='learning rate (default: [0.05, 0.1])')
parser.add_argument(
    '--randbeta',
    type=float,
    default=0,
    help='probability of switching beta (defaut: 0)')

# Network arguments
parser.add_argument(
    '--size_tab',
    nargs = '+',
    type=int,
    default=[10],
    metavar='ST',
    help='tab of layer sizes (default: [10])')
parser.add_argument(
    '--discrete',
    action='store_true',
    default=False,
    help='discrete-time dynamics (default: False)')

# Fundamental hyperparameters
################################################################################
parser.add_argument(
    '--N1',
    type=int,
    default=100,
    metavar='N1',
    help='number of time steps in the forward pass (default: 100)')
parser.add_argument(
    '--N2',
    type=int,
    default=25,
    metavar='N2',
    help='number of time steps in the backward pass (default: 25)')
parser.add_argument(
    '--max-Q',
    type=float,
    default=1.0,
    help='Maximum out charge per time bin')
parser.add_argument(
    '--beta',
    type=float,
    default=1.0,
    metavar='BETA',
    help='nudging parameter (default: 1)')
parser.add_argument(
    '--n-dynamic',
    type=float,
    default=3.5,
    help='Number of time steps to decay by 1/e')
parser.add_argument(
    '--n-trace',
    type=float,
    default=1.8,
    help='Number of time steps to decay by 1/e')
################################################################################

parser.add_argument(
    '--activation-function',
    type=str,
    default='sigm',
    metavar='ACTFUN',
    help='activation function (default: sigmoid)')
parser.add_argument(
    '--no-clamp',
    action='store_true',
    default=False,
    help='clamp neurons between 0 and 1 (default: True)')
parser.add_argument(
    '--learning-rule',
    type=str,
    default='ep',
    metavar='LR',
    help='learning rule (ep/vf, default: ep)')
parser.add_argument(
    '--cep',
    action='store_true',
    default=False,
    help='continual ep/vf (default: False)')
# parser.add_argument(
#     '--angle',
#     type=float,
#     default=0,
#     help='initial angle between forward and backward weights(defaut: 0)')

#other arguments
parser.add_argument(
    '--action',
    type=str,
    default='train',
    help='action to execute (default: train)')
parser.add_argument(
    '--device-label',
    type=int,
    default=0,
    help='selects cuda device (default 0, -1 to select )')
parser.add_argument(
    '--debug-cep',
    action='store_true',
    default=False,
    help='debug cep (default: False)')
parser.add_argument(
    '--seed',
    nargs = '+',
    type=int,
    default=42,
    metavar='SEED',
    help='seed (default: None')
# parser.add_argument(
#     '--angle-grad',
#     action='store_true',
#     default=False,
#     help='computes initial angle between EP updates and BPTT gradients (default: False)')
parser.add_argument(
    '--use-bias',
    action='store_true',
    default=False,
    help='computes initial angle between EP updates and BPTT gradients (default: False)')
parser.add_argument(
    '--no-rhop',
    action='store_true',
    default=False,
    help='computes initial angle between EP updates and BPTT gradients (default: False)')
parser.add_argument(
    '--update-rule',
    type=str,
    default='cep',
    help='set which learning rule to use')
parser.add_argument(
    '--no-reset',
    action='store_true',
    default=False,
    help='reset weights for each batch')
parser.add_argument(
    '--plain-data',
    action='store_true',
    default=False,
    help='reset weights for each batch')
parser.add_argument(
    '--trace-decay',
    type=float,
    default=None,
    help='decay factor for traces')
parser.add_argument(
    '--directory',
    type=str,
    default='output',
    help='select learning rate')
parser.add_argument(
    '--load',
    action='store_true',
    help='if set, loads network from directory')
parser.add_argument(
    '--spiking',
    action='store_true',
    help='if set, uses spikes for dynamics')

# Whatever unit step is in will be the same for tau-dynamic and tau-trace.

# Non-fundamental arguments
parser.add_argument(
    '--T1',
    type=float,
    default=None,
    help='Phase 1 duration')
parser.add_argument(
    '--T2',
    type=float,
    default=None,
    help='Phase 1 duration')
parser.add_argument(
    '--max-fr',
    type=float,
    default=None,
    help='maximum activity / firing rate')
parser.add_argument(
    '--step',
    type=float,
    default=1.0,
    help='time step size')
parser.add_argument(
    '--tau-dynamic',
    type=float,
    default=None,
    help='time constant for dynamics')
parser.add_argument(
    '--tau-trace',
    type=float,
    default=None,
    help='decay factor for traces')
parser.add_argument(
    '--dt',
    type=float,
    default=None,
    metavar='DT',
    help='time discretization (default: 0.2)')
parser.add_argument(
    '--spike-height',
    type=float,
    default=None,
    help='sets height of a spike')


args = parser.parse_args()

# Keep ration of N2 to n_dynamic fixed
# args.n_dynamic=0.233*args.N2
# args.n_trace=0.12*args.N2


# Set steps/durations for both phases
use_time_variables=True
if use_time_variables:
    args.N1 = round(args.T1/args.step) # step should divide T1
    args.N2 = round(args.T2/args.step) # step should divide T2
    args.max_Q = args.max_fr*args.step
    args.spike_height=args.max_fr*args.step
    args.n_dynamic=args.tau_dynamic/args.step
    if args.update_rule == 'stdp' or args.update_rule =='nonspikingstdp':
        args.n_trace=args.tau_trace/args.step


# if args.T1==None:
#     args.T1=args.N1*args.step
# else:
#     assert args.N1==None, "Cannot set both N1 and T1"
#     assert args.T1>args.step, "T1 must be at least one time step"
#     args.N1 = int(args.T1/args.step)
#
# if args.T2==None:
#     args.T2=args.N2*args.step
# else:
#     assert args.N2==None, "Cannot set both N2 and T2"
#     assert args.T2>args.step, "T2 must be at least one time step"
#     args.N2 = int(args.T2/args.step)
#
# if args.max_fr==None:
#     args.max_fr=args.max_Q/args.step
# else:
#     assert args.max_Q==None, "Cannot set both max_fr and max_Q"
#     args.max_Q=args.max_fr*args.step
#
# if args.tau_dynamic==None:
#     args.tau_dynamic = args.n_dynamic*args.step
# else:
#     assert args.n_dynamic==None, "Cannot set both n_dynamic and tau_dynamic"
#     args.n_dynamic=args.tau_dynamics/args.step
#
# if args.tau_trace==None:
#     args.tau_trace = args.n_trace*args.step
# else:
#     assert args.n_trace==None, "Cannot set both n_trace and tau_trace"
#     args.n_trace=args.tau_trace/args.step
# if args.spike_height==None: # spiking_height = max_Q
#     args.spike_height = args.step*args.max_fr

if args.dt==None:
    args.dt = 1-np.exp(-1./args.n_dynamic)
    print("dt = ",args.dt)

if args.trace_decay==None:
    args.trace_decay=np.exp(-1./args.n_trace)
    print("trace decay = ",args.trace_decay)

##### Just making sure #####
# Should these go back in??????????????????
# args.max_Q=1.0
# args.spike_height=1.0


# New this should create consistency as we change the number of steps
# if args.step==None:
#     # args.step=12.8/args.N2
#     args.step=12./args.N2

# Discrete versions of times constants

#
# if args.max_fr==None:
#     args.max_fr = 1.0/args.step


# if not not args.seed:
#     torch.manual_seed(args.seed[0])
set_seed(args.seed)

batch_size = args.batch_size
batch_size_test = args.test_batch_size

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class ReshapeTransformTarget:
    def __init__(self, number_classes):
        self.number_classes = number_classes

    def __call__(self, target):
        target=torch.tensor(target).unsqueeze(0).unsqueeze(1)
        target_onehot=torch.zeros((1,self.number_classes))
        return target_onehot.scatter_(1, target, 1).squeeze(0)



mnist_transforms=[torchvision.transforms.ToTensor(),ReshapeTransform((-1,))]

train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=True, download=True,
                     transform=torchvision.transforms.Compose(mnist_transforms),
                     target_transform=ReshapeTransformTarget(10)),
batch_size = args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=False, download=True,
                     transform=torchvision.transforms.Compose(mnist_transforms),
                     target_transform=ReshapeTransformTarget(10)),
batch_size = args.test_batch_size, shuffle=True)


if  args.activation_function == 'sigm':
    def rho(x):
        return 1/(1+torch.exp(-(4*(x-0.5))))
    def rhop(x):
        return 4*torch.mul(rho(x), 1 -rho(x))

elif args.activation_function == 'hardsigm':
    def rho(x):
        return x.clamp(min = 0).clamp(max = 1)
    def rhop(x):
        return ((x >= 0) & (x <= 1)).float()

elif args.activation_function == 'tanh':
    def rho(x):
        return torch.tanh(x)
    def rhop(x):
        return 1 - torch.tanh(x)**2


if __name__ == '__main__':

    input_size = 28

    #Build the net
    pkl_path = args.directory+'/net'


    if args.load and os.path.exists(pkl_path):
        with open(pkl_path,'rb') as pkl_file:
            net = pickle.load(pkl_file)
    else:
        if  (not args.discrete) & (args.learning_rule == 'vf') :
            net = VFcont(args)

        if (not args.discrete) & (args.learning_rule == 'ep') :
            net = EPcont(args)

        elif (args.discrete) & (args.learning_rule == 'vf'):
            net = VFdisc(args)

        elif (args.discrete) & (args.learning_rule == 'ep'):
            net = EPdisc(args)

        elif args.learning_rule == 'stdp':
            net = SNN(args)

    if args.action == 'train':

        # #create path
        # BASE_PATH, name = createPath(args)
        #
        # #save hyperparameters
        # createHyperparameterfile(BASE_PATH, name, args)

        # Create pickle path

        # Create csv file
        csv_path = args.directory+"/results.csv"
        if not args.load:
            # csv_file = open(csv_path,'a',newline='')
            # csv_writer = csv.write(csvf)
            fieldnames = ['learning_rule','update_rule','beta','dt','N1','N2']
            with open(csv_path,'w',newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(fieldnames)
                csv_writer.writerow([args.learning_rule,args.update_rule,args.beta,args.dt,args.N1,args.N2])
        # #compute initial angle between EP update and BPTT gradient
        # if args.angle_grad:
        #     batch_idx, (example_data, example_targets) = next(enumerate(train_loader))
        #     if net.cuda:
        #         example_data, example_targets = example_data.to(net.device), example_targets.to(net.device)
        #     x = example_data
        #     target = example_targets
        #     nS, dS, dT, _ = compute_nSdSdT(net, x, target)
        #     nT = compute_nT(net, x, target)
        #     theta_T = compute_angleGrad(nS, dS, nT, dT)
        #     results_dict_angle = {'theta_T': theta_T}
        #     print('Initial angle between total EP update and total BPTT gradient: {:.2f} degrees'.format(theta_T))


        #train with EP
        error_train_tab = []
        error_test_tab = []

        start_time = datetime.datetime.now()

        # for epoch in range(net.current_epoch, args.epochs):
        while net.current_epoch<args.epochs:
            epoch=net.current_epoch
            error_train = train(net, train_loader, epoch, args.learning_rule)
            error_train_tab.append(error_train)

            error_test = evaluate(net, test_loader,learning_rule=args.learning_rule)
            error_test_tab.append(error_test) ;
            results_dict = {'error_train_tab' : error_train_tab, 'error_test_tab' : error_test_tab,
                            'elapsed_time': datetime.datetime.now() - start_time}

            # if args.angle_grad:
                # results_dict.update(results_dict_angle)

            with open(csv_path,'a+',newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([error_train, error_test])

            #  Increment epoch and save network
            net.current_epoch += 1
            pkl_path = args.directory+'/net'
            with open(pkl_path,'wb') as pkl_file:
                pickle.dump(net,pkl_file)
