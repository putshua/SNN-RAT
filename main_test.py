import argparse
import os
import sys
from models.VGG import *
import data_loaders
from functions import *
from utils import val
from models import *
import attack
import copy
import torch
import json

parser = argparse.ArgumentParser()
# just use default setting
parser.add_argument('-j','--workers',default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=64, type=int,metavar='N',help='mini-batch size')
parser.add_argument('-sd', '--seed',default=42,type=int,help='seed for initializing training.')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset', default='cifar10',type=str,help='dataset')
parser.add_argument('-arch','--model', default='vgg11', type=str,help='model')
parser.add_argument('-T','--time', default=8, type=int, metavar='N',help='snn simulation time')
parser.add_argument('-id', '--identifier', type=str, help='model statedict identifier')
parser.add_argument('-config', '--config', default='', type=str,help='test configuration file')

# training configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')

# adv atk configuration
parser.add_argument('-atk','--attack',default='',type=str,help='attack')
parser.add_argument('-eps','--eps',default=8,type=float,metavar='N',help='attack eps')
parser.add_argument('-atk_m','--attack_mode',default='', type=str,help='attack mode')

# only pgd
parser.add_argument('-alpha','--alpha',default=2.55/1,type=float,metavar='N',help='pgd attack alpha')
parser.add_argument('-steps','--steps',default=7,type=int,metavar='N',help='pgd attack steps')
parser.add_argument('-bb','--bbmodel',default='',type=str,help='black box model') # vgg11_clean_l2[0.000500]bb
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    if args.dataset.lower() == 'cifar10':
        use_cifar10 = True
        num_labels = 10
    elif args.dataset.lower() == 'cifar100':
        use_cifar10 = False
        num_labels = 100
    elif args.dataset.lower() == 'svhn':
        num_labels = 10

    log_dir = '%s-results'% (args.dataset)

    model_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = get_logger(os.path.join(log_dir, '%s.log'%(args.identifier+args.suffix)))
    logger.info('start testing!')

    seed_all(args.seed)
    if 'cifar' in args.dataset.lower():
        _, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=use_cifar10)
    elif args.dataset.lower() == 'svhn':
        _, val_dataset, znorm = data_loaders.build_svhn()
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if 'vgg_wobn' in args.model.lower():
        model = VGG_woBN(args.model.lower(), args.time, num_labels, znorm)
    elif 'vgg' in args.model.lower():
        model = VGG(args.model.lower(), args.time, num_labels, znorm)
    elif 'wideresnet' in args.model.lower():
        model = WideResNet(args.model.lower(), args.time, num_labels, znorm)
    else:
        raise AssertionError("model not supported")

    model.set_simulation_time(args.time)
    model.to(device)

    # have bb model
    if len(args.bbmodel) > 0:
        bbmodel = copy.deepcopy(model)
        bbstate_dict = torch.load(os.path.join(model_dir, args.bbmodel+'.pth'), map_location=torch.device('cpu'))
        bbmodel.load_state_dict(bbstate_dict, strict=False)
    else:
        bbmodel = None

    if len(args.config) > 0:
        with open(args.config+'.json', 'r') as f:
            config = json.load(f)
    else:
        config = [{}]
    for atk_config in config:
#         logger.info(json.dumps(atk_config))
        for arg in atk_config.keys():
            setattr(args, arg, atk_config[arg])
        if 'bb' in atk_config.keys() and atk_config['bb']:
            atkmodel = bbmodel
        else:
            atkmodel = model

        if args.attack_mode == 'bptt':
            ff = BPTT_attack
        elif 'rate' in args.attack_mode:
            ff = BPTR_attack
        else:
            ff = Act_attack

        if args.attack.lower() == 'fgsm':
            atk = attack.FGSM(atkmodel, forward_function=ff, eps=args.eps / 255, T=args.time)
        elif args.attack.lower() == 'pgd':
            atk = attack.PGD(atkmodel, forward_function=ff, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, T=args.time)
        elif args.attack.lower() == 'gn':
            atk = attack.GN(atkmodel, forward_function=ff, eps=args.eps / 255, T=args.time)
        else:
            atk = None
        
        state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.to(device)

        acc = val(model, test_loader, device, args.time, atk)
        logger.info(json.dumps(atk_config)+' Test acc={:.3f}'.format(acc))

if __name__ == "__main__":
    main()
