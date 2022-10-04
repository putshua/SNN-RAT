import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from data_loaders import cifar10
from functions import *
import attack
from models import *
from models.VGG import VGG_woBN
from utils import train, val

parser = argparse.ArgumentParser()
# just use default setting
parser.add_argument('-j','--workers',default=8, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=64, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('--optim', default='sgd',type=str,help='optimizer')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar10',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg11',type=str,help='model')
parser.add_argument('-T','--time',default=8, type=int,metavar='N',help='snn simulation time')

# training configuration
parser.add_argument('--epochs',default=200,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-lr','--lr',default=0.1,type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-dev','--device',default='0',type=str,help='device')

# adv training configuration
parser.add_argument('-special','--special', default='l2', type=str, help='[reg, l2]')
parser.add_argument('-beta','--beta',default=5e-4, type=float,help='regulation beta')
parser.add_argument('-atk','--attack',default='', type=str,help='attack')
parser.add_argument('-eps','--eps',default=2, type=float, metavar='N',help='attack eps')
parser.add_argument('-atk_m','--attack_mode',default='', type=str,help='[bptt, bptr, '']')

# only PGD
parser.add_argument('-alpha','--alpha',default=1, type=float, metavar='N', help='pgd attack alpha')
parser.add_argument('-steps','--steps',default=2, type=int, metavar='N', help='pgd attack steps')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    if args.dataset.lower() == 'cifar10':
        num_labels = 10
        train_dataset, val_dataset, znorm = cifar10()

    log_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_all(args.seed)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

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

    if args.attack_mode == 'bptt':
        ff = BPTT_attack
    elif args.attack_mode == 'bptr':
        ff = BPTR_attack
    else:
        ff = None

    if args.attack.lower() == 'fgsm':
        atk = attack.FGSM(model, forward_function=ff, eps=args.eps / 255, T=args.time)
    elif args.attack.lower() == 'pgd':
        atk = attack.PGD(model, forward_function=ff, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, T=args.time)
    elif args.attack.lower() == 'gn':
        atk = attack.GN(model, eps=args.eps / 255)
    else:
        atk = None

    criterion = nn.CrossEntropyLoss().to(device)

    if args.optim.lower() == 'adam' and args.special == 'l2':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.beta)
    elif args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'sgd' and args.special == 'l2':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.beta)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0

    identifier = args.model
    if atk is not None:
        identifier += '_%s[%f][%s]' %(atk.__class__.__name__, atk.eps, args.attack_mode)
    else:
        identifier += '_clean'

    identifier += '_%s[%f]'%(args.special, args.beta)
    identifier += args.suffix

    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))
    logger.info('start training!')
    
    for epoch in range(args.epochs):
        loss, acc = train(model, device, train_loader, criterion, optimizer, args.time, atk=atk, beta=args.beta, parseval=(args.special == 'reg'))
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc))
        scheduler.step()
        tmp = val(model, test_loader, device, args.time)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}\n'.format(epoch , args.epochs, tmp))

        if best_acc < tmp:
            best_acc = tmp
            torch.save(model.state_dict(), os.path.join(log_dir, '%s.pth'%(identifier)))

    logger.info('Best Test acc={:.3f}'.format(best_acc))

if __name__ == "__main__":
    main()
