"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import time
import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser
from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from helper.util import adjust_learning_rate
from distiller_zoo import TTM, WTTM, DistillKL, CRDLoss, ITLoss, DIST, ABKL, LS, DKD
from helper.loops import train_distill as train, validate


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'resnet18', 'resnet34', 'resnet50',
                                 'resnet101', 'resnet152'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd',
                        choices=['kd', 'ttm', 'wttm', 'crd', 'itrd', 'dist', 'ab', 'ls', 'dkd'])
    parser.add_argument('--add', type=str, default='kd', choices=['kd', 'ttm', 'wttm'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--seed', type=int, default=0, help='seed id, set to 0 if do not want to fix the seed')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for additional loss')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for main loss')

    # KD distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # TTM and WTTM distillation
    parser.add_argument('--ttm_l', type=float, default=1, help='exponent for TTM and WTTM distillation')

    # ITRD distillation
    parser.add_argument('--lambda_corr', type=float, default=2.0, help='correlation loss weight')
    parser.add_argument('--lambda_mutual', type=float, default=1.0, help='mutual information loss weight')
    parser.add_argument('--alpha_it', type=float, default=1.50, help='Renyis alpha')

    # DIST distillation
    parser.add_argument('--dist_beta', type=float, default=1, help='weight for inter loss')
    parser.add_argument('--dist_gamma', type=float, default=1, help='weight for intra loss')
    parser.add_argument('--dist_tau', type=float, default=4, help='temperature for DIST distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    ## AB divergence
    parser.add_argument('--ab_alpha', type=float, default=0.5, help='weight balance for hardness-concentration')
    parser.add_argument('--ab_beta', type=float, default=0.5, help='weight balance for confidence-concentration')

    # DKD distillation
    parser.add_argument('--dkd_alpha', type=float, default=1, help='weight balance for target-class distillation')
    parser.add_argument('--dkd_beta', type=float, default=8, help='weight balance for non target class distillation')
    parser.add_argument('--dkd_warmup', type=float, default=20, help='the nums of epoch for warmup')

    opt = parser.parse_args()

    # set different learning rate from these 3 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/student_model'
    opt.log_pth = './save/student_log'

    if opt.alpha > 0:
        opt.model_path = './save/student_model_add_' + opt.add
        opt.log_pth = './save/student_log_add_' + opt.add

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_temp:{}_l:{}_r:{}_a:{}_b:{}_alpha_{}_beta_{}_{}'.format(
        opt.model_s, opt.model_t, opt.dataset, opt.distill, opt.kd_T, opt.ttm_l,
        opt.gamma, opt.alpha, opt.beta, opt.ab_alpha, opt.ab_beta, opt.seed)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_key = 'S:{}_T:{}_{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill)

    opt.log_folder = os.path.join(opt.log_pth, opt.log_key)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    best_acc = 0

    opt = parse_option()

    if opt.seed:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        opt.n_cls = n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()

    if opt.add == 'kd':
        criterion_div = DistillKL(opt.kd_T)
    elif opt.add == 'ttm':
        criterion_div = TTM(opt.ttm_l)
    elif opt.add == 'wttm':
        criterion_div = WTTM(opt.ttm_l)
    else:
        raise NotImplementedError(opt.add)

    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'dkd':
        criterion_kd = DKD(opt)
    elif opt.distill == 'ls':
        criterion_kd = LS(opt)
    elif opt.distill == 'ab':
        criterion_kd = ABKL(opt)
    elif opt.distill == 'ttm':
        criterion_kd = TTM(opt)
    elif opt.distill == 'wttm':
        criterion_kd = WTTM(opt.ttm_l)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'itrd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = ITLoss(opt)
        module_list.append(criterion_kd)
        trainable_list.append(criterion_kd)
        module_list.append(criterion_kd.embed)
        trainable_list.append(criterion_kd.embed)
    elif opt.distill == 'dist':
        criterion_kd = DIST(opt.dist_beta, opt.dist_gamma, opt.dist_tau)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # additional loss
    criterion_list.append(criterion_kd)  # distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        module_list.to(device)
        criterion_list.to(device)
        if not opt.seed:
            cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, module_list, criterion_list, opt, is_Teacher=True)
    print('teacher accuracy: ', teacher_acc)

    # log file
    log_fname = os.path.join(opt.log_folder, '{experiment}.txt'.format(experiment=opt.model_name))

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss, train_entropy_temp, train_entropy_no_temp = train(
            epoch, train_loader, module_list, criterion_list, optimizer, opt
        )
        time2 = time.time()
        print('Epoch {}, Total Time: {:.2f}'.format(epoch, time2 - time1))

        test_acc, test_acc_top5, test_loss = validate(val_loader, module_list, criterion_list, opt)

        # Logging train, test metrics, and entropies
        with open(log_fname, 'a') as log:
            log.write(f"Epoch {epoch}, "
                      f"Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, "
                      f"Train Entropy (Temp): {train_entropy_temp:.4f}, Train Entropy (No Temp): {train_entropy_no_temp:.4f}, "
                      f"Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}\n")

        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('Saving the best model!')
            torch.save(state, save_file)

        # Print current epoch metrics
        print(f"Epoch {epoch}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, "
              f"Train Entropy (Temp): {train_entropy_temp:.4f}, Train Entropy (No Temp): {train_entropy_no_temp:.4f}, "
              f"Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    print('Best Accuracy:', best_acc)

    # save best accuracy
    with open(log_fname, 'a') as log:
        log.write('best: ' + str(best_acc.cpu().numpy()) + '\n')

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
