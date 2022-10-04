import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import data_loaders
from functions import seed_all, get_logger, RateBP_attack, BPTT_attack
import attack
from models import *
from models.VGG import VGG_woBN
from utils import train_mix, val

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers',default=8, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=64, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('--optim', default='sgd',type=str,help='model')
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
parser.add_argument('-special','--special', default='l2', type=str, help='[parseval, l2]')
parser.add_argument('-beta','--beta',default=5e-4, type=float,help='regulation beta')
parser.add_argument('-eps','--eps',default=2, type=float, metavar='N',help='attack eps')

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

    log_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_all(args.seed)
    if 'cifar' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=use_cifar10)
    elif args.dataset.lower() == 'svhn':
        train_dataset, val_dataset, znorm = data_loaders.build_svhn()
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

    # atk = [attack.FGSM(model, forward_function=BPTT_attack, eps=args.eps / 255, T=args.time),
    #        attack.FGSM(model, forward_function=RateBP_attack, eps=args.eps / 255, T=args.time),
    #        attack.RFGSM(model, forward_function=BPTT_attack, eps=args.eps / 255, T=args.time),
    #        attack.RFGSM(model, forward_function=RateBP_attack, eps=args.eps / 255, T=args.time)]

    atk = [attack.FGSM(model, forward_function=RateBP_attack, eps=args.eps / 255, T=args.time),
           attack.RFGSM(model, forward_function=RateBP_attack, eps=args.eps / 255, T=args.time)]

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

    # IMPORTANT<<<<<<<<<<<<< modifed
    identifier = args.model

    identifier += '_mix[%f]' %(args.eps)

    identifier += '_%s[%f]'%(args.special, args.beta)
    identifier += args.suffix

    parseval = (args.special == 'parseval')

    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))
    logger.info('start training!')
    
    for epoch in range(args.epochs):
        loss, acc = train_mix(model, device, train_loader, criterion, optimizer, args.time, atk_list=atk, beta=args.beta, parseval=parseval)
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
